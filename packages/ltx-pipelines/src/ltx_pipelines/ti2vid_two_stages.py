import logging
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
    video_conditionings_by_replacing_latent,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


def blend_audio_with_crossfade(
    input_audio: torch.Tensor,
    generated_audio: torch.Tensor,
    input_duration_samples: int,
    crossfade_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    """Blend input audio with generated audio using crossfade at the transition point.

    Args:
        input_audio: Audio from input video, shape (samples,) or (channels, samples).
        generated_audio: Generated audio for full duration, same shape convention.
        input_duration_samples: Number of samples from input video duration.
        crossfade_samples: Number of samples for crossfade region.
        sample_rate: Audio sample rate (for logging).

    Returns:
        Blended audio tensor with input audio preserved and crossfade to generated.
    """
    # Ensure both are 1D (mono) for simplicity - handle stereo if needed
    if input_audio.ndim == 2:
        input_audio = input_audio.mean(dim=0)  # Convert to mono
    if generated_audio.ndim == 2:
        generated_audio = generated_audio.mean(dim=0)  # Convert to mono

    total_samples = generated_audio.shape[0]

    # If input is longer than output, just use generated
    if input_duration_samples >= total_samples:
        return generated_audio

    # Create output tensor
    output = torch.zeros_like(generated_audio)

    # Ensure crossfade doesn't exceed available samples
    crossfade_start = max(0, input_duration_samples - crossfade_samples)
    crossfade_end = min(input_duration_samples + crossfade_samples, total_samples)
    actual_crossfade = crossfade_end - crossfade_start

    # Copy input audio up to crossfade start
    input_samples_available = min(input_audio.shape[0], crossfade_start)
    output[:input_samples_available] = input_audio[:input_samples_available]

    # Apply crossfade in the transition region
    if actual_crossfade > 0:
        # Create crossfade weights
        fade_out = torch.linspace(1.0, 0.0, actual_crossfade, device=generated_audio.device)
        fade_in = torch.linspace(0.0, 1.0, actual_crossfade, device=generated_audio.device)

        # Get the audio segments for crossfade
        input_fade_start = min(crossfade_start, input_audio.shape[0])
        input_fade_end = min(crossfade_end, input_audio.shape[0])
        input_fade_len = input_fade_end - input_fade_start

        if input_fade_len > 0:
            # Pad input audio if needed
            input_crossfade_segment = torch.zeros(actual_crossfade, device=generated_audio.device)
            input_crossfade_segment[:input_fade_len] = input_audio[input_fade_start:input_fade_end]
        else:
            input_crossfade_segment = torch.zeros(actual_crossfade, device=generated_audio.device)

        gen_crossfade_segment = generated_audio[crossfade_start:crossfade_end]

        # Blend
        output[crossfade_start:crossfade_end] = (
            input_crossfade_segment * fade_out + gen_crossfade_segment * fade_in
        )

    # Copy generated audio after crossfade
    output[crossfade_end:] = generated_audio[crossfade_end:]

    return output


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        image_crf: float | None = None,
        skip_cleanup: bool = False,
        video_extend_path: str | None = None,
        video_extend_strength: float = 1.0,
        audio_extend_mode: str = "generate",
        audio_crossfade_duration: float = 0.5,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        # Extract input audio for blending if requested
        input_audio = None
        input_video_frames = 0
        if video_extend_path and audio_extend_mode == "blend":
            input_audio = decode_audio_from_file(video_extend_path, device=self.device)
            # Count input video frames to determine blend point
            import av
            with av.open(video_extend_path) as container:
                video_stream = next(s for s in container.streams if s.type == "video")
                input_video_frames = video_stream.frames or 0
                # If frames count unavailable, estimate from duration
                if input_video_frames == 0 and video_stream.duration and video_stream.time_base:
                    duration_sec = float(video_stream.duration * video_stream.time_base)
                    input_video_frames = int(duration_sec * frame_rate)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.stage_1_model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        if not skip_cleanup:
            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            image_crf=image_crf,
        )

        # Add video extension conditioning if provided
        if video_extend_path:
            video_conds = video_conditionings_by_replacing_latent(
                video_path=video_extend_path,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                strength=video_extend_strength,
                start_frame_idx=0,
                max_frames=None,
            )
            stage_1_conditionings.extend(video_conds)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            del transformer
            cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            image_crf=image_crf,
        )

        # Add video extension conditioning for stage 2 as well
        if video_extend_path:
            video_conds = video_conditionings_by_replacing_latent(
                video_path=video_extend_path,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                strength=video_extend_strength,
                start_frame_idx=0,
                max_frames=None,
            )
            stage_2_conditionings.extend(video_conds)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )

        # Blend input audio with generated audio if requested
        if input_audio is not None and audio_extend_mode == "blend":
            # Calculate blend point based on input video duration
            input_duration_sec = float(input_video_frames) / float(frame_rate)
            input_duration_samples = int(input_duration_sec * AUDIO_SAMPLE_RATE)
            crossfade_samples = int(audio_crossfade_duration * AUDIO_SAMPLE_RATE)

            decoded_audio = blend_audio_with_crossfade(
                input_audio=input_audio,
                generated_audio=decoded_audio,
                input_duration_samples=input_duration_samples,
                crossfade_samples=crossfade_samples,
                sample_rate=AUDIO_SAMPLE_RATE,
            )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
