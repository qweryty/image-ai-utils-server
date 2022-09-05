# This file was created by contributors of Diffusers library.
# The original code can be found here:
# https://github.com/huggingface/diffusers/blob/main/examples/inference/image_to_image.py
import inspect
from typing import List, Optional, Union, Callable, Awaitable

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, \
    UNet2DConditionModel, StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def preprocess(image: Image.Image) -> torch.FloatTensor:
    image = image.convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask: Image.Image) -> torch.FloatTensor:
    mask = mask.convert('L')
    w, h = mask.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    if w < h:
        h = int(h / (w / 64))
        w = 64
    else:
        w = int(w / (h / 64))
        h = 64

    mask = mask.resize((w, h), resample=Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)
    return mask


def mask_overlay(
        first: torch.FloatTensor, second: torch.FloatTensor, mask: torch.FloatTensor
) -> torch.FloatTensor:
    return first * (1 - mask) + second * mask


class StableDiffusionUniversalPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )

    def _scale_and_encode(self, image: torch.FloatTensor):
        latents = self.vae.encode(image).sample()
        return 0.18215 * latents

    def _scale_and_decode(self, latents):
        return self.vae.decode(1 / 0.18215 * latents)

    async def text_to_image(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            run_safety_checker: bool = False,
            progress_callback: Optional[Callable[[int, int], Awaitable]] = None
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f'`height` and `width` have to be divisible by 8 but are {height} and {width}.')

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [''] * batch_size, padding='max_length', max_length=max_length, return_tensors='pt'
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}')
            latents = latents.to(self.device)

        # set timesteps
        accepts_offset = 'offset' in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs['offset'] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            if progress_callback is not None:
                await progress_callback(i, len(self.scheduler.timesteps))
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, i, latents, **extra_step_kwargs
                )['prev_sample']
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )['prev_sample']

        # scale and decode the image latents with vae
        image = self._scale_and_decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if run_safety_checker:
            # run safety checker
            safety_cheker_input = self.feature_extractor(
                self.numpy_to_pil(image),
                return_tensors='pt'
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_cheker_input.pixel_values
            )
        else:
            has_nsfw_concept = False

        image = self.numpy_to_pil(image)

        return {'sample': image, 'nsfw_content_detected': has_nsfw_concept}

    async def image_to_image(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        mask: Optional[torch.FloatTensor] = None,
        alpha: Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        run_safety_checker: bool = False,
        progress_callback: Optional[Callable[[int, int], Awaitable]] = None
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')

        if strength < 0 or strength > 1:
            raise ValueError(f'The value of strength should in [0.0, 1.0] but is {strength}')

        # set timesteps
        accepts_offset = 'offset' in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs['offset'] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # encode the init image into latents and scale the latents
        init_latents = self._scale_and_encode(init_image)
        if alpha is not None:
            # Replacing transparent area with noise
            init_latents = mask_overlay(
                init_latents,
                torch.randn(init_latents.shape, generator=generator, device=self.device),
                alpha
            )
        init_latents_orig = init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [''] * batch_size, padding='max_length', max_length=max_length, return_tensors='pt'
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        time_steps = self.scheduler.timesteps[t_start:]
        for i, t in enumerate(time_steps):
            if progress_callback is not None:
                await progress_callback(i, len(time_steps))
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            )['prev_sample']

            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = mask_overlay(init_latents_proper, latents, mask)

        # scale and decode the image latents with vae
        image = self._scale_and_decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if run_safety_checker:
            safety_cheker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors='pt'
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_cheker_input.pixel_values
            )
        else:
            has_nsfw_concept = False

        image = self.numpy_to_pil(image)

        return {'sample': image, 'nsfw_content_detected': has_nsfw_concept}
