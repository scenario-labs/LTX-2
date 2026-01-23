import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
import torch.nn as nn

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def get_submodule_and_parent(root: nn.Module, path: str):
    """
    Returns (parent_module, child_name, child_module)
    where child_module is reachable at `path` from root.
    Supports numeric segments for Sequential/ModuleList.
    """
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]  # Sequential/ModuleList
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        child = parent[int(last)]
    else:
        child = getattr(parent, last)
    return parent, last, child


def set_submodule(root: nn.Module, path: str, new_module: nn.Module):
    parent, last, _ = get_submodule_and_parent(root, path)
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


class MultiLoraLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        # Each entry: dict with CPU tensors always, and optional CUDA cache
        self.adapters: list[dict] = []
        self.enabled: list[bool] = []

    def add_adapter_cpu(self, A_cpu, B_cpu, scale=1.0, enabled=False, pin_memory=False):
        if A_cpu is None or B_cpu is None:
            self.adapters.append({
                "A_cpu": None,
                "B_cpu": None,
                "scale": float(scale),
                "A_gpu": None,
                "B_gpu": None,
                "gpu_dtype": None,
                "gpu_device": None,
            })
            self.enabled.append(bool(enabled))
            return

        A_cpu = A_cpu.contiguous()
        B_cpu = B_cpu.contiguous()
        self.adapters.append({
            "A_cpu": A_cpu,
            "B_cpu": B_cpu,
            "scale": float(scale),
            "A_gpu": None,
            "B_gpu": None,
            "gpu_dtype": None,
            "gpu_device": None,
        })
        self.enabled.append(bool(enabled))

    def _materialize_to_gpu(self, idx: int):
        entry = self.adapters[idx]
        if entry["A_cpu"] is None or entry["B_cpu"] is None:
            return
        """Move adapter idx to the base weight device/dtype if not already there."""
        entry = self.adapters[idx]
        dev = self.base.weight.device
        dt = self.base.weight.dtype

        if (
            entry["A_gpu"] is not None
            and entry["B_gpu"] is not None
            and entry["gpu_device"] == dev
            and entry["gpu_dtype"] == dt
        ):
            return  # already good

        A = entry["A_cpu"].to(device=dev, dtype=dt, non_blocking=True)
        B = entry["B_cpu"].to(device=dev, dtype=dt, non_blocking=True)

        entry["A_gpu"] = A
        entry["B_gpu"] = B
        entry["gpu_device"] = dev
        entry["gpu_dtype"] = dt

    def _evict_from_gpu(self, idx: int):
        """Drop CUDA copies (free VRAM). CPU tensors remain."""
        entry = self.adapters[idx]
        entry["A_gpu"] = None
        entry["B_gpu"] = None
        entry["gpu_device"] = None
        entry["gpu_dtype"] = None

    def set_enabled(self, idx: int, enabled: bool, offload_when_disabled: bool = True):
        if not (0 <= idx < len(self.enabled)):
            return
        self.enabled[idx] = enabled
        if not enabled and offload_when_disabled:
            self._evict_from_gpu(idx)

    def forward(self, x):
        out = self.base(x)
        for i, on in enumerate(self.enabled):
            if not on:
                continue
            entry = self.adapters[i]
            if entry["A_cpu"] is None or entry["B_cpu"] is None:
                continue
            if entry["A_gpu"] is None or entry["B_gpu"] is None:
                self._materialize_to_gpu(i)
            A = entry["A_gpu"]
            B = entry["B_gpu"]
            out = out + ((x @ A.t()) @ B.t()) * entry["scale"]
        return out


def set_lora_enabled(model: nn.Module, adapter_idx: int, enabled: bool, offload_when_disabled: bool = True):
    for m in model.modules():
        if isinstance(m, MultiLoraLinear):
            m.set_enabled(adapter_idx, enabled, offload_when_disabled=offload_when_disabled)


def enable_only_lora(model: nn.Module, adapter_idx: int | None):
    # disable all
    # (assumes all layers have same number of adapters; true if you patch consistently)
    for m in model.modules():
        if isinstance(m, MultiLoraLinear):
            for i in range(len(m.enabled)):
                m.set_enabled(i, False, offload_when_disabled=True)

    torch.cuda.empty_cache()

    # enable selected
    if adapter_idx is not None and adapter_idx >= 0:
        set_lora_enabled(model, adapter_idx, True)

    torch.cuda.empty_cache()


def patch_only_affected_linears(
    model: nn.Module,
    lora_sd: dict,  # can be CPU state dict
    affected_modules: list[str],
    strength: float,
    adapter_idx: int,
    default_enabled: bool = False,
):
    for prefix in affected_modules:
        _, _, mod = get_submodule_and_parent(model, prefix)

        if isinstance(mod, MultiLoraLinear):
            wrapped = mod
        else:
            if not isinstance(mod, nn.Linear):
                continue
            wrapped = MultiLoraLinear(mod)
            set_submodule(model, prefix, wrapped)

        # ensure adapter slots exist up to adapter_idx
        while len(wrapped.adapters) <= adapter_idx:
            wrapped.add_adapter_cpu(None, None, scale=0.0, enabled=False)

        key_a = f"{prefix}.lora_A.weight"
        key_b = f"{prefix}.lora_B.weight"
        if key_a not in lora_sd or key_b not in lora_sd:
            # leave the padded empty slot
            continue

        A_cpu = lora_sd[key_a]
        B_cpu = lora_sd[key_b]

        # overwrite the placeholder slot
        wrapped.adapters[adapter_idx] = {
            "A_cpu": A_cpu.contiguous(),
            "B_cpu": B_cpu.contiguous(),
            "scale": float(strength),
            "A_gpu": None,
            "B_gpu": None,
            "gpu_dtype": None,
            "gpu_device": None,
        }
        wrapped.enabled[adapter_idx] = default_enabled


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)
        model_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device)

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            sd = model_state_dict.sd
            if dtype is not None:
                sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.sd.items()}
            meta_model.load_state_dict(sd, strict=False, assign=True)
            return self._return_model(meta_model, device)

        # Load LoRA[0] (fused) on GPU (or CPU—GPU is fine since you fuse immediately)
        lora0_sd = self.load_sd(
            [self.loras[0].path],
            sd_ops=self.loras[0].sd_ops,
            registry=self.registry,
            device=device,
        )

        # Load runtime LoRAs on CPU so they don't sit in VRAM
        runtime_lora_sds = [
            self.load_sd(
                [lora.path],
                sd_ops=lora.sd_ops,
                registry=self.registry,
                device=torch.device("cpu"),
            )
            for lora in self.loras[1:]
        ]

        # Rebuild lists to match your later code expectations
        lora_state_dicts = [lora0_sd, *runtime_lora_sds]

        lora_sd_and_strengths = [
            LoraStateDictWithStrength(sd, strength)
            for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
        ]
        final_sd = apply_loras(
            model_sd=model_state_dict,
            lora_sd_and_strengths=[lora_sd_and_strengths[0]],
            dtype=dtype,
            destination_sd=model_state_dict if isinstance(self.registry, DummyRegistry) else None,
        )
        meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
        model = self._return_model(meta_model, device)

        _, affected_modules = apply_loras(
            model_sd=model_state_dict,
            lora_sd_and_strengths=lora_sd_and_strengths[1:],
            dtype=dtype,
            destination_sd=None,
            return_affected=True,
        )

        for runtime_idx, (lora_sd, strength) in enumerate(zip(lora_state_dicts[1:], lora_strengths[1:], strict=True)):
            patch_only_affected_linears(
                model,
                lora_sd.sd,
                affected_modules,
                strength=strength,
                adapter_idx=runtime_idx,
                default_enabled=False,  # start off
            )

        return model
