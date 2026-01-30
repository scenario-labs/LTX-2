"""Audio conditioning by latent replacement."""

import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class AudioConditionByLatent(ConditioningItem):
    """
    Conditions audio generation by injecting latents.

    Replaces audio tokens in the latent state and sets denoise strength
    according to the strength parameter. Similar to VideoConditionByLatentIndex
    but for the entire audio duration.
    """

    def __init__(self, latent: torch.Tensor, strength: float):
        """Initialize audio conditioning.

        Args:
            latent: Audio latent tensor, shape (batch, channels, frames, mel_bins).
            strength: Conditioning strength (0-1). 1.0 = fully conditioned (no denoising).
        """
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """Apply audio conditioning by replacing tokens and setting denoise mask.

        Args:
            latent_state: Current latent state to modify.
            latent_tools: Tools for patchifying/unpatchifying latents.

        Returns:
            Modified latent state with audio conditioning applied.
        """
        import logging
        logger = logging.getLogger(__name__)

        tokens = latent_tools.patchifier.patchify(self.latent)
        num_tokens = tokens.shape[1]
        total_tokens = latent_state.latent.shape[1]
        logger.info(f"[AudioCond.apply_to] Input latent: {self.latent.shape}")
        logger.info(f"[AudioCond.apply_to] Patchified tokens: {tokens.shape}")
        logger.info(f"[AudioCond.apply_to] Replacing {num_tokens}/{total_tokens} tokens, strength={self.strength}")

        latent_state = latent_state.clone()

        # Replace tokens from the beginning
        latent_state.latent[:, :num_tokens] = tokens
        latent_state.clean_latent[:, :num_tokens] = tokens
        # Set denoise mask: 0 = full denoising, 1 = no denoising
        # strength=1.0 means no denoising (preserve input)
        mask_value = 1.0 - self.strength
        latent_state.denoise_mask[:, :num_tokens] = mask_value
        logger.info(f"[AudioCond.apply_to] Set denoise_mask[:, :{num_tokens}] = {mask_value}")

        return latent_state
