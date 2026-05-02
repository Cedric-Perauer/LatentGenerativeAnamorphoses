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

import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from ...models import AutoencoderKLFlux2, Flux2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import Flux2ImageProcessor
from .pipeline_output import Flux2PipelineOutput
from .system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I

from .lod_new import (
    create_vertical_flip_warp,
    create_identity_warp,
    create_circular_rotation_warp,
    create_conic_mirror_warp,
    create_jigsaw_warp,
    view_simple,
    view_lod,
    soften_inverse_conic,
    LaplacianPyramid,
    Laplacian2Gaussian,
    blend_pyramids,
    blend_pyramids_masked,
    masked_blend,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Flux2Pipeline

        >>> pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=50, guidance_scale=2.5).images[0]
        >>> image.save("flux.png")
        ```
"""

UPSAMPLING_MAX_IMAGE_SIZE = 768**2


# Adapted from
# https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L68
def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
    images: list[PIL.Image.Image, list[list[PIL.Image.Image]]] | None = None,
):
    """
    Format a batch of text prompts into the conversation format expected by apply_chat_template. Optionally, add images
    to the input.

    Args:
        prompts: List of text prompts
        system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
        images (optional): List of images to add to the input.

    Returns:
        List of conversations, where each conversation is a list of message dicts
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]
    else:
        assert len(images) == len(prompts), "Number of images must match number of prompts"
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            ]
            for _ in cleaned_txt
        ]

        for i, (el, images) in enumerate(zip(messages, images)):
            # optionally add the images per batch element.
            if images is not None:
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image_obj} for image_obj in images],
                    }
                )
            # add the text.
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


# Adapted from
# https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L49C5-L66C19
def _validate_and_process_images(
    images: list[list[PIL.Image.Image]] | list[PIL.Image.Image],
    image_processor: Flux2ImageProcessor,
    upsampling_max_image_size: int,
) -> list[list[PIL.Image.Image]]:
    # Simple validation: ensure it's a list of PIL images or list of lists of PIL images
    if not images:
        return []

    # Check if it's a list of lists or a list of images
    if isinstance(images[0], PIL.Image.Image):
        # It's a list of images, convert to list of lists
        images = [[im] for im in images]

    # potentially concatenate multiple images to reduce the size
    images = [[image_processor.concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in images]

    # cap the pixels
    images = [
        [image_processor._resize_if_exceeds_area(img_i, upsampling_max_image_size) for img_i in img_i]
        for img_i in images
    ]
    return images


# Taken from
# https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/sampling.py#L251
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Flux2Pipeline(DiffusionPipeline):
    r"""
    The Flux2 pipeline for text-to-image generation.

    Reference: [https://bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2)

    Args:
        transformer ([`Flux2Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Mistral3ForConditionalGeneration`]):
            [Mistral3ForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration)
        tokenizer (`AutoProcessor`):
            Tokenizer of class
            [PixtralProcessor](https://huggingface.co/docs/transformers/en/model_doc/pixtral#transformers.PixtralProcessor).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        transformer: Flux2Transformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        self.system_message = SYSTEM_MESSAGE
        self.system_message_upsampling_t2i = SYSTEM_MESSAGE_UPSAMPLING_T2I
        self.system_message_upsampling_i2i = SYSTEM_MESSAGE_UPSAMPLING_I2I
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

    @staticmethod
    def _get_mistral_3_small_prompt_embeds(
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        prompt: str | list[str],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        system_message: str = SYSTEM_MESSAGE,
        hidden_states_layers: list[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: torch.Tensor | None = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _prepare_image_ids(
        image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):
        r"""
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (list[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """

        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def upsample_prompt(
        self,
        prompt: str | list[str],
        images: list[PIL.Image.Image, list[list[PIL.Image.Image]]] = None,
        temperature: float = 0.15,
        device: torch.device = None,
    ) -> list[str]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = self.text_encoder.device if device is None else device

        # Set system message based on whether images are provided
        if images is None or len(images) == 0 or images[0] is None:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I

        # Validate and process the input images
        if images:
            images = _validate_and_process_images(images, self.image_processor, self.upsampling_max_image_size)

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message, images=images)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device, self.text_encoder.dtype)

        # Generate text using the model's generate method
        generated_ids = self.text_encoder.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
        )

        # Decode only the newly generated tokens (skip input tokens)
        # Extract only the generated portion
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        upsampled_prompt = self.tokenizer.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return upsampled_prompt

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (10, 20, 30),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_mistral_3_small_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(imagge_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    # =========================================================================
    # Anamorphic helper methods (ported from Flux1 pipeline)
    # =========================================================================

    def flip_tensor(self, noise_sample, mode='horizontal'):
        if mode == 'horizontal':
            return torch.flip(noise_sample, [3])
        elif mode == 'vertical':
            return torch.flip(noise_sample, [2])
        elif mode == '90flip':
            return torch.rot90(noise_sample, 1, [2, 3])
        elif mode in ('90rot', '135rot', '180rot', 'conic'):
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
        elif mode in ('90rot', '135rot', '180rot', 'conic'):
            return self.apply_laplacian_warp(noise_sample, transform_type=mode, inverse=True)
        elif mode == 'jigsaw':
            return self.apply_laplacian_warp(noise_sample, transform_type='jigsaw', inverse=True)
        else:
            return noise_sample

    def apply_laplacian_warp(self, image, transform_type, inverse=False):
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
        elif transform_type == "conic":
            r_in_ratio = getattr(self, '_conic_r_in_ratio', 0.15)
            r_out_ratio = getattr(self, '_conic_r_out_ratio', 0.95)
            warp, mask = create_conic_mirror_warp(
                h, w,
                r_in_ratio=r_in_ratio,
                r_out_ratio=r_out_ratio,
                inverse=inverse,
            )
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

        if transform_type == "conic" and not inverse:
            warped = view_lod(image, warp, leveln=5, padding_mode='border')
        else:
            warped = view_simple(image, warp)

        if (transform_type == "conic" and inverse and mask is not None
                and getattr(self, "_apply_conic_soften", False)):
            warped = soften_inverse_conic(warped, mask)
        self._current_warp_mask = mask

        return warped.to(image.dtype)

    def lwp_blend(self, img1, img2, alpha=0.5, leveln=5, use_pyramids=True, mask=None):
        while img1.ndim > 4:
            img1 = img1.squeeze(0)
        while img2.ndim > 4:
            img2 = img2.squeeze(0)

        # Per-level mask-weighted Laplacian blending for partial-view warps,
        # then hard-composite with a binary mask so the silhouette boundary
        # stays strict (no coarse-level mask blur leaking img2 across).
        if mask is not None and use_pyramids:
            lp1 = LaplacianPyramid(img1, leveln)
            lp2 = LaplacianPyramid(img2, leveln)
            blended_lp = blend_pyramids_masked(lp1, lp2, mask, alpha=alpha)
            gp = Laplacian2Gaussian(blended_lp)
            blended = gp[0]
            hard_mask = (mask > 0.5).to(blended.dtype)
            if hard_mask.dim() == 4 and hard_mask.shape[1] != blended.shape[1]:
                hard_mask = hard_mask.expand(-1, blended.shape[1], -1, -1)
            return hard_mask * blended + (1.0 - hard_mask) * img1

        if mask is not None:
            blended = masked_blend(img1, img2, mask, alpha=alpha)
            hard_mask = (mask > 0.5).to(blended.dtype)
            if hard_mask.dim() == 4 and hard_mask.shape[1] != blended.shape[1]:
                hard_mask = hard_mask.expand(-1, blended.shape[1], -1, -1)
            return hard_mask * blended + (1.0 - hard_mask) * img1

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
    # Flux2-specific pack/unpack helpers for the anamorphic denoising loop
    # =========================================================================

    def _unpack_to_spatial(self, latents, latent_ids):
        """Unpack Flux2 packed latents (B, seq, C) to spatial (B, C_orig, H, W) via BN denorm + unpatchify."""
        spatial = self._unpack_latents_with_ids(latents, latent_ids)  # (B, C*4, H//2, W//2)
        return spatial

    def _pack_to_sequence(self, latents):
        """Pack spatial latents (B, C*4, H//2, W//2) back to Flux2 format (B, seq, C*4)."""
        return self._pack_latents(latents)

    def _bn_denormalize(self, latents):
        """Apply batch norm denormalization for VAE decode."""
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * latents_bn_std + latents_bn_mean

    def _bn_normalize(self, latents):
        """Apply batch norm normalization after VAE encode."""
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
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

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_image2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds2: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (10, 20, 30),
        transform_type: Optional[str] = None,
        jigsaw_seed: Optional[int] = 42,
        time_travel: Optional[int] = 1,
        time_travel_range: Optional[List[int]] = [20, 80],
        denoise_last: bool = False,
        denoise_last_steps: int = 5,
        vis_intermediate: bool = False,
        lwp: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for anamorphic generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt for the first view of the anamorphic image.
            prompt_image2 (`str` or `List[str]`, *optional*):
                The prompt for the second view (revealed after transformation).
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Embedded guidance scale for Flux2.
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
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompts (dual prompts for anamorphoses)
        prompt_embeds1, text_ids1 = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        prompt_embeds2, text_ids2 = self.encode_prompt(
            prompt=prompt_image2,
            prompt_embeds=prompt_embeds2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, latent_ids = self.prepare_latents(
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
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas_input,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # Store jigsaw seed
        self._jigsaw_seed = jigsaw_seed

        # Compute time travel step range
        start_time_travel_step = int(num_inference_steps * (time_travel_range[0] / 100))
        end_time_travel_step = int(num_inference_steps * (time_travel_range[1] / 100))
        time_travel_step_range = list(range(start_time_travel_step, end_time_travel_step))
        print(f'Time travel will happen between steps {start_time_travel_step} and {end_time_travel_step}')

        # Second scheduler for the second latent stream
        self.scheduler2 = copy.deepcopy(self.scheduler)

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

                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # Forward pass for view 1
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds1,
                        txt_ids=text_ids1,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                    # Forward pass for view 2
                    if not denoise_one:
                        noise_pred2 = self.transformer(
                            hidden_states=latents2,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds2,
                            txt_ids=text_ids2,
                            img_ids=latent_ids,
                            joint_attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                    # One-step clean prediction (jump to the end)
                    clean_sample1 = self.scheduler.step_to_the_end(noise_pred, t, latents)[0]
                    if not denoise_one:
                        clean_sample2 = self.scheduler2.step_to_the_end(noise_pred2, t, latents2)[0]

                    # Unpack to spatial: (B, seq, C) -> (B, C*4, H//2, W//2)
                    clean_sample1_spatial = self._unpack_to_spatial(clean_sample1, latent_ids)
                    if not denoise_one:
                        clean_sample2_spatial = self._unpack_to_spatial(clean_sample2, latent_ids)

                    # BN denormalize + unpatchify + VAE decode to image space
                    clean_latent1_denorm = self._bn_denormalize(clean_sample1_spatial)
                    clean_latent1_unpatch = self._unpatchify_latents(clean_latent1_denorm)
                    if not denoise_one:
                        clean_latent2_denorm = self._bn_denormalize(clean_sample2_spatial)
                        clean_latent2_unpatch = self._unpatchify_latents(clean_latent2_denorm)

                    clean_img_pred1 = self.vae.decode(clean_latent1_unpatch, return_dict=False)[0]
                    if not denoise_one:
                        clean_img_pred2 = self.vae.decode(clean_latent2_unpatch, return_dict=False)[0]

                    if vis_intermediate:
                        image = self.image_processor.postprocess(clean_img_pred1, output_type=output_type)[0]
                        image.save(f'intermediate_imgs/image1_{i:03d}.png')

                    # Re-encode through VAE + patchify + BN normalize
                    rencoded_latents1 = retrieve_latents(self.vae.encode(clean_img_pred1), generator)
                    rencoded_latents1 = self._patchify_latents(rencoded_latents1)
                    rencoded_latents1 = self._bn_normalize(rencoded_latents1)
                    if not denoise_one:
                        rencoded_latents2 = retrieve_latents(self.vae.encode(clean_img_pred2), generator)
                        rencoded_latents2 = self._patchify_latents(rencoded_latents2)
                        rencoded_latents2 = self._bn_normalize(rencoded_latents2)

                    # Correction term (in patchified spatial space)
                    correction_term = clean_sample1_spatial - rencoded_latents1
                    if not denoise_one:
                        correction_term2 = clean_sample2_spatial - rencoded_latents2

                    ### LWP related -------------------

                    # Transform and blend in image space
                    if not denoise_one:
                        if lwp:
                            clean_img_pred2 = self.apply_laplacian_warp(clean_img_pred2, transform_type=transform_type)
                            blend_mask = getattr(self, '_current_warp_mask', None)
                        else:
                            clean_img_pred2 = self.flip_tensor(clean_img_pred2, mode=transform_type)
                            blend_mask = None
                        blended_img = self.lwp_blend(clean_img_pred1, clean_img_pred2, use_pyramids=lwp_use_pyramids, mask=blend_mask)
                    else:
                        blended_img = clean_img_pred1

                    # Blend correction terms (in unpatchified image-like space for warping)
                    if not denoise_one:
                        # Unpatchify correction terms for warping
                        corr1_unpatch = self._unpatchify_latents(correction_term)
                        corr2_unpatch = self._unpatchify_latents(correction_term2)
                        if lwp:
                            corr2_unpatch = self.apply_laplacian_warp(corr2_unpatch, transform_type=transform_type)
                            blend_mask = getattr(self, '_current_warp_mask', None)
                        else:
                            corr2_unpatch = self.flip_tensor(corr2_unpatch, mode=transform_type)
                            blend_mask = None
                        blended_correction_unpatch = self.lwp_blend(corr1_unpatch, corr2_unpatch, use_pyramids=lwp_use_pyramids, mask=blend_mask)
                    else:
                        blended_correction_unpatch = self._unpatchify_latents(correction_term)

                    ### Second loop (warp back) -------------------

                    warped_img1 = copy.deepcopy(blended_img)
                    if not denoise_one:
                        if lwp:
                            warped_img2 = self.apply_laplacian_warp(copy.deepcopy(blended_img), transform_type=transform_type, inverse=True)
                        else:
                            warped_img2 = self.inverse_flip_tensor(copy.deepcopy(blended_img), mode=transform_type)

                    # Encode warped images back through VAE + patchify + BN normalize
                    warped_latents1 = retrieve_latents(self.vae.encode(warped_img1), generator)
                    warped_latents1 = self._patchify_latents(warped_latents1)
                    warped_latents1 = self._bn_normalize(warped_latents1)
                    if not denoise_one:
                        warped_latents2 = retrieve_latents(self.vae.encode(warped_img2), generator)
                        warped_latents2 = self._patchify_latents(warped_latents2)
                        warped_latents2 = self._bn_normalize(warped_latents2)

                    # Apply correction (warp correction back too)
                    correct1_warp_unpatch = copy.deepcopy(blended_correction_unpatch)
                    correct1_warp = self._patchify_latents(correct1_warp_unpatch)
                    if not denoise_one:
                        if lwp:
                            correct2_warp_unpatch = self.apply_laplacian_warp(copy.deepcopy(blended_correction_unpatch), transform_type=transform_type, inverse=True)
                        else:
                            correct2_warp_unpatch = self.inverse_flip_tensor(copy.deepcopy(blended_correction_unpatch), mode=transform_type)
                        correct2_warp = self._patchify_latents(correct2_warp_unpatch)

                    clean_latent = warped_latents1 + correct1_warp
                    if not denoise_one:
                        clean_latent2 = warped_latents2 + correct2_warp

                    # Pack back to sequence format for scheduler
                    clean_latent_packed = self._pack_to_sequence(clean_latent)
                    if not denoise_one:
                        clean_latent2_packed = self._pack_to_sequence(clean_latent2)

                    # Latent interpolation (scheduler step)
                    latents = self.scheduler.latent_interpolation(clean_latent_packed, latents, t)
                    if not denoise_one:
                        latents2 = self.scheduler2.latent_interpolation(clean_latent2_packed, latents2, t)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 7. Decode final output
        if output_type == "latent":
            image = latents
        else:
            latents_spatial = self._unpack_to_spatial(latents, latent_ids)
            latents_spatial = self._bn_denormalize(latents_spatial)
            latents_spatial = self._unpatchify_latents(latents_spatial)

            image = self.vae.decode(latents_spatial, return_dict=False)[0]
            self._apply_conic_soften = True
            image2 = self.inverse_flip_tensor(image, mode=transform_type)
            self._apply_conic_soften = False
            image = self.image_processor.postprocess(image, output_type=output_type)
            image2 = self.image_processor.postprocess(image2, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return image[0], image2[0]
