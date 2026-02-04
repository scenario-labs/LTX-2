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
    but for audio with optional positioning via place_at_end.
    """

    def __init__(self, latent: torch.Tensor, strength: float, place_at_end: bool = False):
        """Initialize audio conditioning.

        Args:
            latent: Audio latent tensor, shape (batch, channels, frames, mel_bins).
            strength: Conditioning strength (0-1). 1.0 = fully conditioned (no denoising).
            place_at_end: If True, place audio at end of output (for backward extend).
                If False (default), place at beginning (for forward extend).
        """
        self.latent = latent
        self.strength = strength
        self.place_at_end = place_at_end

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """Apply audio conditioning by replacing tokens and setting denoise mask.

        Args:
            latent_state: Current latent state to modify.
            latent_tools: Tools for patchifying/unpatchifying latents.

        Returns:
            Modified latent state with audio conditioning applied.
        """
        tokens = latent_tools.patchifier.patchify(self.latent)
        num_tokens = tokens.shape[1]
        total_tokens = latent_state.latent.shape[1]

        latent_state = latent_state.clone()

        # Calculate start position based on placement
        if self.place_at_end:
            start = total_tokens - num_tokens
        else:
            start = 0
        stop = start + num_tokens

        latent_state.latent[:, start:stop] = tokens
        latent_state.clean_latent[:, start:stop] = tokens
        # Set denoise mask: 0 = full denoising, 1 = no denoising
        # strength=1.0 means no denoising (preserve input)
        latent_state.denoise_mask[:, start:stop] = 1.0 - self.strength

        return latent_state
