# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for Latent Generative Anamorphoses: dual-prompt, time-travel,
# Laplacian Weighted Pooling (LWP), and transformation-based illusion generation.

import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...image_processor import VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import FluxPipelineOutput

from .lod_new import (
    create_vertical_flip_warp,
    create_identity_warp,
    create_circular_rotation_warp,
    create_jigsaw_warp,
    view_simple,
    LaplacianPyramid,
    Laplacian2Gaussian,
    blend_pyramids,
    masked_blend,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image1, image2 = pipe(
        ...     prompt="a pop art of albert einstein",
        ...     prompt_image2="a pop art of marilyn monroe",
        ...     transform_type="vertical",
        ...     num_inference_steps=28,
        ... )
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FluxPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux pipeline for text-to-image generation, modified for Latent Generative Anamorphoses.

    Supports dual prompts, time-travel denoising, Laplacian Weighted Pooling (LWP),
    and transformation-based visual illusions (flips, rotations, jigsaw).

    Reference: https://huggingface.co/black-forest-labs/FLUX.1-dev

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are 2x2 packed, so image_processor uses vae_scale_factor * 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    # =========================================================================
    # Text Encoding
    # =========================================================================

    def _get_t5_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt=1,
        device=None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        prompt_2=None,
        device=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        max_sequence_length=512,
        lora_scale=None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # =========================================================================
    # Latent Handling (pack/unpack for Flux's 2x2 patchification)
    # =========================================================================

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression and packing requires divisibility by 2
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def prepare_latents_unpacked(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Prepare latents in unpacked (4D spatial) format for VAE encode/decode operations."""
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # =========================================================================
    # Anamorphoses Helper Methods
    # =========================================================================

    def flip_tensor(self, noise_sample, mode='horizontal'):
        if mode == 'horizontal':
            return torch.flip(noise_sample, [3])
        elif mode == 'vertical':
            return torch.flip(noise_sample, [2])
        elif mode == '90flip':
            return torch.rot90(noise_sample, 1, [2, 3])
        elif mode in ('90rot', '135rot', '180rot'):
            return self.apply_laplacian_warp(noise_sample, transform_type=mode, inverse=False)
        elif mode == 'jigsaw':
            return self.apply_laplacian_warp(noise_sample, transform_type='jigsaw', inverse=False)
        else:
            return noise_sample

    def inverse_flip_tensor(self, noise_sample, mode='horizontal'):
        if mode == 'horizontal':
            return torch.flip(noise_sample, [3])
        elif mode == 'vertical':
            return torch.flip(noise_sample, [2])
        elif mode == '90flip':
            return torch.rot90(noise_sample, -1, [2, 3])
        elif mode in ('90rot', '135rot', '180rot'):
            return self.apply_laplacian_warp(noise_sample, transform_type=mode, inverse=True)
        elif mode == 'jigsaw':
            return self.apply_laplacian_warp(noise_sample, transform_type='jigsaw', inverse=True)
        else:
            return noise_sample

    def apply_laplacian_warp(self, image, transform_type, inverse=False):
        """
        Apply warping to an image using the specified transform type.

        Args:
            image: Image tensor (B, C, H, W)
            transform_type: Type of transform ("vertical", "horizontal", "90rot", "135rot", "180rot", "jigsaw")
            inverse: Whether to apply inverse warp

        Returns:
            Warped image tensor
        """
        while image.ndim > 4:
            image = image.squeeze(0)

        b, c, h, w = image.shape
        mask = None

        if transform_type == "90flip":
            self._current_warp_mask = None
            k = -1 if inverse else 1
            return torch.rot90(image, k, [2, 3])

        if transform_type == "vertical":
            warp = create_vertical_flip_warp(h, w)

        elif transform_type == "horizontal":
            v_coords = torch.linspace(0, 1, h).view(-1, 1).expand(h, w)
            u_coords = torch.linspace(0, 1, w).view(1, -1).expand(h, w)
            warp_u = 1.0 - u_coords
            warp_v = v_coords
            warp = torch.stack([warp_u, warp_v], dim=0).unsqueeze(0)
            third_channel = torch.zeros(1, 1, h, w)
            warp = torch.cat([warp, third_channel], dim=1)

        elif transform_type == "90rot":
            angle = -90.0 if inverse else 90.0
            warp, mask = create_circular_rotation_warp(h, w, angle, radius_ratio=0.45)

        elif transform_type == "135rot":
            angle = -135.0 if inverse else 135.0
            warp, mask = create_circular_rotation_warp(h, w, angle, radius_ratio=0.45)

        elif transform_type == "180rot":
            angle = -180.0 if inverse else 180.0
            warp, mask = create_circular_rotation_warp(h, w, angle, radius_ratio=0.45)

        elif transform_type == "jigsaw":
            jigsaw_seed = getattr(self, '_jigsaw_seed', 42)
            cache = getattr(self, "_jigsaw_warp_cache", {})
            cache_key = (h, w, jigsaw_seed)
            if cache_key not in cache:
                warp_fwd = create_jigsaw_warp(h, w, seed=jigsaw_seed, inverse=False)
                warp_inv = create_jigsaw_warp(h, w, seed=jigsaw_seed, inverse=True)
                cache[cache_key] = (warp_fwd, warp_inv)
                self._jigsaw_warp_cache = cache
            warp = cache[cache_key][1 if inverse else 0]

        else:
            warp = create_identity_warp(h, w)

        warp = warp.to(image.device, dtype=torch.float32)
        if mask is not None:
            mask = mask.to(image.device, dtype=image.dtype)

        warped = view_simple(image, warp)
        self._current_warp_mask = mask

        return warped.to(image.dtype)

    def lwp(self, img1, img2, alpha=0.5, leveln=5, use_pyramids=True, mask=None):
        """
        Laplacian Weighted Pooling with combined averaging.

        From the paper (equations 12-15):
        - Standard average: avg(x,y) = (x + y) / 2
        - Value-weighted average: vavg(x,y) = (|x|*x + |y|*y) / (|x| + |y|)
        - Final blend: z = avg + alpha(vavg - avg)
        """
        while img1.ndim > 4:
            img1 = img1.squeeze(0)
        while img2.ndim > 4:
            img2 = img2.squeeze(0)

        if mask is not None:
            return masked_blend(img1, img2, mask, alpha=alpha)

        if use_pyramids:
            lp1 = LaplacianPyramid(img1, leveln)
            lp2 = LaplacianPyramid(img2, leveln)
            blended_lp = blend_pyramids(lp1, lp2, alpha=alpha)
            gp = Laplacian2Gaussian(blended_lp)
            return gp[0]
        else:
            stacked = torch.stack([img1, img2])
            avg_result = torch.nanmean(stacked, dim=0)

            img1_safe = torch.where(torch.isnan(img1), torch.zeros_like(img1), img1)
            img2_safe = torch.where(torch.isnan(img2), torch.zeros_like(img2), img2)

            abs_img1 = torch.abs(img1_safe)
            abs_img2 = torch.abs(img2_safe)

            numerator = abs_img1 * img1_safe + abs_img2 * img2_safe
            denominator = abs_img1 + abs_img2

            epsilon = 1e-8
            vavg_result = numerator / torch.clamp(denominator, min=epsilon)

            blended = avg_result + alpha * (vavg_result - avg_result)
            return blended

    # =========================================================================
    # Flux-specific pack/unpack helpers for the denoising loop
    # =========================================================================

    def _unpack_to_spatial(self, latents, height, width):
        """Unpack Flux packed latents (B, seq, C*4) to spatial (B, C, H, W) for VAE operations."""
        return self._unpack_latents(latents, height, width, self.vae_scale_factor)

    def _pack_from_spatial(self, latents):
        """Pack spatial latents (B, C, H, W) back to Flux format (B, seq, C*4)."""
        b, c, h, w = latents.shape
        return self._pack_latents(latents, b, c, h, w)

    # =========================================================================
    # Main Generation
    # =========================================================================

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_image2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_image2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        transform_type: Optional[str] = None,
        jigsaw_seed: Optional[int] = 42,
        time_travel: Optional[int] = 1,
        time_travel_range: Optional[List[int]] = [20, 80],
        denoise_last: bool = False,
        denoise_last_steps: int = 5,
        vis_intermediate: bool = False,
        lwp: bool = False,
        lwp_alpha: float = 0.5,
    ):
        r"""
        Function invoked when calling the pipeline for anamorphic generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt for the first view of the anamorphic image.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt to be sent to the T5 encoder. If not defined, `prompt` is used.
            prompt_image2 (`str` or `List[str]`, *optional*):
                The prompt for the second view (revealed after transformation).
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guidance scale for Flux (passed to transformer, not used for CFG).
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt used for true classifier-free guidance when `true_cfg_scale > 1`.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1, enables true CFG (batched cond/uncond passes, like SD3) on top of the
                embedded guidance. This restores the per-step corrective pull toward each view's
                prompt that the anamorphosis blending relies on.
            transform_type (`str`, *optional*):
                Transformation type: "vertical", "horizontal", "90rot", "135rot", "180rot", "jigsaw".
            jigsaw_seed (`int`, *optional*, defaults to 42):
                Random seed for jigsaw puzzle permutation.
            time_travel (`int`, *optional*, defaults to 1):
                Number of backward re-noising steps for time-travel denoising.
            time_travel_range (`List[int]`, *optional*, defaults to [20, 80]):
                Percentage range [start, end] of steps where time-travel is applied.
            denoise_last (`bool`, *optional*, defaults to False):
                If True, final steps only denoise the first view (no blending).
            denoise_last_steps (`int`, *optional*, defaults to 5):
                Number of final steps for single-view denoising.
            vis_intermediate (`bool`, *optional*, defaults to False):
                If True, save intermediate images for debugging.
            lwp (`bool`, *optional*, defaults to False):
                If True, use Laplacian Weighted Pooling for blending.

        Examples:

        Returns:
            Tuple of (image1, image2): First view and transformed view as PIL Images.
        """

        lwp_use_pyramids = True if lwp else False
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # 3. Encode prompts (dual prompts for anamorphoses)
        (
            prompt_embeds1,
            pooled_prompt_embeds1,
            text_ids1,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        (
            prompt_embeds2,
            pooled_prompt_embeds2,
            text_ids2,
        ) = self.encode_prompt(
            prompt=prompt_image2,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        do_true_cfg = true_cfg_scale > 1.0
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt if negative_prompt is not None else "",
                prompt_2=negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            # Per-view negative for the second stream (defaults to the shared negative)
            if negative_prompt_image2 is not None:
                (
                    negative_prompt_embeds2,
                    negative_pooled_prompt_embeds2,
                    _,
                ) = self.encode_prompt(
                    prompt=negative_prompt_image2,
                    prompt_2=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )
            else:
                negative_prompt_embeds2 = negative_prompt_embeds
                negative_pooled_prompt_embeds2 = negative_pooled_prompt_embeds

        # 4. Prepare latent variables
        # Flux uses in_channels // 4 for the unpacked channel count
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds1.dtype,
            device,
            generator,
            latents=None,
        )

        latents2, _ = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds2.dtype,
            device,
            generator,
            latents=None,
        )

        initial_noise = copy.deepcopy(latents)
        initial_noise2 = copy.deepcopy(latents2)

        # 5. Prepare timesteps
        sigmas_input = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas_input = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas_input,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle embedded guidance (Flux uses guidance embedding, not CFG)
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Store jigsaw seed
        self._jigsaw_seed = jigsaw_seed

        # Compute time travel step range
        start_time_travel_step = int(num_inference_steps * (time_travel_range[0] / 100))
        end_time_travel_step = int(num_inference_steps * (time_travel_range[1] / 100))
        time_travel_step_range = list(range(start_time_travel_step, end_time_travel_step))
        print(f'Time travel will happen between steps {start_time_travel_step} and {end_time_travel_step}')

        # Second scheduler for the second latent stream
        self.scheduler2 = copy.deepcopy(self.scheduler)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        def transformer_pred(latents_in, prompt_embeds_in, pooled_in, text_ids_in, t_in,
                             neg_embeds=None, neg_pooled=None):
            # Single velocity prediction; with true CFG the cond/uncond passes are batched
            # like the SD3 pipeline and combined via uncond + scale * (cond - uncond).
            if do_true_cfg:
                neg_embeds = neg_embeds if neg_embeds is not None else negative_prompt_embeds
                neg_pooled = neg_pooled if neg_pooled is not None else negative_pooled_prompt_embeds
                latent_model_input = torch.cat([latents_in] * 2)
                encoder_hidden_states = torch.cat([neg_embeds, prompt_embeds_in])
                pooled_projections = torch.cat([neg_pooled, pooled_in])
            else:
                latent_model_input = latents_in
                encoder_hidden_states = prompt_embeds_in
                pooled_projections = pooled_in

            timestep_in = t_in.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
            guidance_in = guidance.expand(latent_model_input.shape[0]) if guidance is not None else None

            pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_in / 1000,
                guidance=guidance_in,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids_in,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            if do_true_cfg:
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + true_cfg_scale * (pred_text - pred_uncond)
            return pred

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        self.scheduler2.set_begin_index(0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                time_range = [i]
                cur_time_travel = time_travel
                if cur_time_travel > i - 1:
                    cur_time_travel = i - 1

                denoise_one = False
                if denoise_last:
                    if i >= num_inference_steps - denoise_last_steps:
                        denoise_one = True

                if i in time_travel_step_range and time_travel > 0:
                    # Re-noise backward: from step i to step (i - time_travel)
                    latents = self.scheduler.scale_noise(
                        latents,
                        timesteps[i],
                        noise=initial_noise,
                        time_travel=cur_time_travel
                    )
                    if not denoise_one:
                        latents2 = self.scheduler2.scale_noise(
                            latents2,
                            timesteps[i],
                            noise=initial_noise2,
                            time_travel=cur_time_travel
                        )

                    # Denoise forward: from (i - cur_time_travel) to i
                    time_range = list(range(i - cur_time_travel, i + 1))

                for i in time_range:
                    t = timesteps[i]

                    # Forward pass for view 1
                    noise_pred = transformer_pred(latents, prompt_embeds1, pooled_prompt_embeds1, text_ids1, t)

                    # Forward pass for view 2
                    if not denoise_one:
                        noise_pred2 = transformer_pred(
                            latents2, prompt_embeds2, pooled_prompt_embeds2, text_ids2, t,
                            neg_embeds=negative_prompt_embeds2 if do_true_cfg else None,
                            neg_pooled=negative_pooled_prompt_embeds2 if do_true_cfg else None,
                        )

                    # Step 4: one-step clean prediction (jump to the end)
                    clean_sample1 = self.scheduler.step_to_the_end(noise_pred, t, latents)[0]
                    if not denoise_one:
                        clean_sample2 = self.scheduler2.step_to_the_end(noise_pred2, t, latents2)[0]

                    # Unpack for VAE operations
                    clean_sample1_spatial = self._unpack_to_spatial(clean_sample1, height, width)
                    if not denoise_one:
                        clean_sample2_spatial = self._unpack_to_spatial(clean_sample2, height, width)

                    # Step 5: decode to image space
                    clean_latent1 = (clean_sample1_spatial / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    if not denoise_one:
                        clean_latent2 = (clean_sample2_spatial / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                    clean_img_pred1 = self.vae._decode(clean_latent1, return_dict=False)[0]
                    if not denoise_one:
                        clean_img_pred2 = self.vae._decode(clean_latent2, return_dict=False)[0]

                    if vis_intermediate:
                        image = self.image_processor.postprocess(clean_img_pred1, output_type=output_type)[0]
                        image.save(f'intermediate_imgs/image1_{i:03d}.png')

                    # Re-encode through VAE
                    rencoded_latents1 = retrieve_latents(self.vae.encode(clean_img_pred1), generator)
                    if not denoise_one:
                        rencoded_latents2 = retrieve_latents(self.vae.encode(clean_img_pred2), generator)

                    rencoded_latents1 = (rencoded_latents1 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    if not denoise_one:
                        rencoded_latents2 = (rencoded_latents2 - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                    # Step 6: correction term
                    correction_term = clean_sample1_spatial - rencoded_latents1
                    if not denoise_one:
                        correction_term2 = clean_sample2_spatial - rencoded_latents2

                    ### LWP related -------------------

                    # Step 8: transform and blend
                    if not denoise_one:
                        if lwp:
                            clean_img_pred2 = self.apply_laplacian_warp(clean_img_pred2, transform_type=transform_type)
                            blend_mask = getattr(self, '_current_warp_mask', None)
                        else:
                            clean_img_pred2 = self.flip_tensor(clean_img_pred2, mode=transform_type)
                            blend_mask = None
                        blended_img = self.lwp(clean_img_pred1, clean_img_pred2, alpha=lwp_alpha, use_pyramids=lwp_use_pyramids, mask=blend_mask)
                    else:
                        blended_img = clean_img_pred1

                    # Step 9: blend correction terms
                    if not denoise_one:
                        if lwp:
                            correction_term2 = self.apply_laplacian_warp(correction_term2, transform_type=transform_type)
                            blend_mask = getattr(self, '_current_warp_mask', None)
                        else:
                            correction_term2 = self.flip_tensor(correction_term2, mode=transform_type)
                            blend_mask = None
                        blended_correction = self.lwp(correction_term, correction_term2, alpha=lwp_alpha, use_pyramids=lwp_use_pyramids, mask=blend_mask)
                    else:
                        blended_correction = correction_term

                    ### Second loop (steps 11-15) -------------------

                    warped_img1 = copy.deepcopy(blended_img)
                    if not denoise_one:
                        if lwp:
                            warped_img2 = self.apply_laplacian_warp(copy.deepcopy(blended_img), transform_type=transform_type, inverse=True)
                        else:
                            warped_img2 = self.inverse_flip_tensor(copy.deepcopy(blended_img), mode=transform_type)

                    # Encode warped images back through VAE
                    warped_latents1 = retrieve_latents(self.vae.encode(warped_img1), generator)
                    if not denoise_one:
                        warped_latents2 = retrieve_latents(self.vae.encode(warped_img2), generator)

                    warped_latents1 = (warped_latents1 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    if not denoise_one:
                        warped_latents2 = (warped_latents2 - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                    # Apply correction
                    correct1_warp = copy.deepcopy(blended_correction)
                    if not denoise_one:
                        if lwp:
                            correct2_warp = self.apply_laplacian_warp(copy.deepcopy(blended_correction), transform_type=transform_type, inverse=True)
                        else:
                            correct2_warp = self.inverse_flip_tensor(copy.deepcopy(blended_correction), mode=transform_type)

                    clean_latent = warped_latents1 + correct1_warp
                    if not denoise_one:
                        clean_latent2 = warped_latents2 + correct2_warp

                    # Re-pack spatial latents back to Flux packed format
                    clean_latent_packed = self._pack_from_spatial(clean_latent)
                    if not denoise_one:
                        clean_latent2_packed = self._pack_from_spatial(clean_latent2)

                    # Latent interpolation (scheduler step)
                    latents = self.scheduler.latent_interpolation(clean_latent_packed, latents, t)
                    if not denoise_one:
                        latents2 = self.scheduler2.latent_interpolation(clean_latent2_packed, latents2, t)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 7. Decode final output
        if output_type == "latent":
            image = latents
        else:
            # Unpack from Flux format to spatial
            latents_spatial = self._unpack_to_spatial(latents, height, width)
            latents_spatial = (latents_spatial / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents_spatial, return_dict=False)[0]
            image2 = self.inverse_flip_tensor(image, mode=transform_type)
            image = self.image_processor.postprocess(image, output_type=output_type)
            image2 = self.image_processor.postprocess(image2, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return image[0], image2[0]
