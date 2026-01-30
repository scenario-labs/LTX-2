import torch

from ltx_core.conditioning.exceptions import ConditioningError
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class AudioConditionByLatentIndex(ConditioningItem):
    """
    Conditions audio generation by injecting latents at a specific latent frame index.
    Replaces tokens in the audio latent state at positions corresponding to latent_idx,
    and sets denoise strength according to the strength parameter.

    This is the audio equivalent of VideoConditionByLatentIndex, enabling audio
    extension/conditioning by freezing the beginning of the audio sequence.
    """

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        """
        Args:
            latent: Audio latent tensor of shape (batch, channels, frames, mel_bins).
            strength: Conditioning strength (0-1). 1.0 = fully conditioned (frozen).
            latent_idx: Starting frame index in the output audio where conditioning begins.
        """
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        # Audio latent shape: (batch, channels, frames, mel_bins)
        cond_batch, cond_channels, _, cond_mel_bins = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_mel_bins = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_mel_bins) != (tgt_batch, tgt_channels, tgt_mel_bins):
            raise ConditioningError(
                f"Can't apply audio conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_mel_bins}). Make sure "
                "the audio and latent have the same batch, channel, and mel_bins dimensions."
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.latent_idx)
        )
        stop_token = start_token + tokens.shape[1]

        latent_state = latent_state.clone()

        latent_state.latent[:, start_token:stop_token] = tokens
        latent_state.clean_latent[:, start_token:stop_token] = tokens
        latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength

        return latent_state
