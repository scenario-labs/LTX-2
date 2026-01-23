import torch
import triton

from ltx_core.loader.kernels import fused_add_round_kernel
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from typing import Iterable

BLOCK_SIZE = 1024


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def calculate_weight_float8_(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    result = fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
    target_weights.copy_(result, non_blocking=True)
    return target_weights


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        product = torch.matmul(lsd.sd[key_b] * coef, lsd.sd[key_a])
        deltas.append(product.to(dtype=dtype, device=device))
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    return torch.sum(torch.stack(deltas, dim=0), dim=0)

def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype,
    destination_sd: StateDict | None = None,
    return_affected: bool = False,
) -> StateDict | tuple[StateDict, list[str]]:
    sd = destination_sd.sd if destination_sd is not None else {}
    size = 0
    device = torch.device("meta")
    inner_dtypes = set()

    affected_weight_keys: list[str] = []
    affected_module_prefixes: set[str] = set()

    for key, weight in model_sd.sd.items():
        if weight is None:
            continue
        if not key.endswith(".weight"):
            # optional: skip non-weight tensors if your SD has them
            continue

        device = weight.device
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype  # you said ignore fp8 path

        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)

        # Record which weights are actually modified by LoRA
        if deltas is not None:
            affected_weight_keys.append(key)
            affected_module_prefixes.add(key[: -len(".weight")])

        if deltas is None:
            if key in sd:
                continue
            out = weight.clone().to(dtype=target_dtype, device=device)
        else:
            # normal add_ path (bf16 etc)
            out = deltas.to(dtype=target_dtype)
            # IMPORTANT: add base weight
            out.add_(weight.to(dtype=out.dtype, device=device))

        sd[key] = out
        inner_dtypes.add(target_dtype)
        size += out.nbytes

    result = destination_sd if destination_sd is not None else StateDict(sd, device, size, inner_dtypes)

    if return_affected:
        # sorted for stable output
        affected = sorted(affected_module_prefixes)
        return result, affected

    return result
