import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional, List, Tuple, Dict
import safetensors.torch as sf
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image

from diffusers_helper.wan_components.wan.distributed.fsdp import shard_model
from diffusers_helper.wan_components.wan.modules.clip import CLIPModel
from diffusers_helper.wan_components.wan.modules.model import WanModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection,UMT5EncoderModel,AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKLWan
from diffusers_helper.wan_components.wan.modules.vae import WanVAE
from diffusers_helper.wan_components.wan.modules.t5 import T5Encoder
from diffusers_helper.wan_components.wan.modules.tokenizers import HuggingfaceTokenizer
from diffusers_helper.wan_components.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from diffusers_helper.wan_components.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanFramePackConfig:
    """Configuration class for WAN FramePack integration"""
    
    def __init__(self):
        # Model parameters
        self.num_train_timesteps = 1000
        self.param_dtype = torch.bfloat16
        self.vae_stride = [4, 8, 8]
        self.patch_size = [1, 2, 2]
        
        # FramePack parameters
        self.latent_window_size = 9
        self.max_video_length = 120.0  # seconds
        self.default_fps = 30
        self.use_teacache = True
        
        # Sampling parameters
        self.default_sampling_steps = 25
        self.default_guidance_scale = 10.0
        self.default_shift = 5.0
        self.sample_neg_prompt = "low quality, worst quality, blurry, distorted"
        
        # Memory management
        self.offload_models = True
        self.quantization_type = 'fp8'  # 'fp8', 'fp16', 'int8'
        self.enable_gradient_checkpointing = True
        
        # Output settings
        self.output_fps = 30
        self.output_crf = 16  # Video compression quality


class FramePackSampler:
    """Implements FramePack's progressive sampling technique for long video generation"""
    
    def __init__(self, latent_window_size: int = 9, vae_stride: List[int] = [4, 8, 8]):
        self.latent_window_size = latent_window_size
        self.vae_stride = vae_stride
        
    def calculate_sections(self, total_second_length: float, fps: int = 30) -> Tuple[int, List[int]]:
        """Calculate the number of sections and padding pattern for generation"""
        total_frames = int(total_second_length * fps)
        frames_per_section = self.latent_window_size * 4
        total_latent_sections = (total_frames) / frames_per_section
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        # FramePack's padding pattern
        if total_latent_sections > 4:
            # Duplicate some items for better quality when sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            latent_paddings = list(reversed(range(total_latent_sections)))
            
        return total_latent_sections, latent_paddings
    
    def prepare_indices(self, latent_padding: int, include_4x: bool = True) -> Dict[str, torch.Tensor]:
        """Prepare indices for clean latents based on padding"""
        latent_padding_size = latent_padding * self.latent_window_size
        
        if include_4x:
            indices = torch.arange(0, sum([1, latent_padding_size, self.latent_window_size, 1, 2, 16])).unsqueeze(0)
            splits = indices.split([1, latent_padding_size, self.latent_window_size, 1, 2, 16], dim=1)
            
            return {
                'clean_latent_indices_pre': splits[0],
                'blank_indices': splits[1],
                'latent_indices': splits[2],
                'clean_latent_indices_post': splits[3],
                'clean_latent_2x_indices': splits[4],
                'clean_latent_4x_indices': splits[5],
                'clean_latent_indices': torch.cat([splits[0], splits[3]], dim=1)
            }
        else:
            indices = torch.arange(0, sum([1, latent_padding_size, self.latent_window_size, 1])).unsqueeze(0)
            splits = indices.split([1, latent_padding_size, self.latent_window_size, 1], dim=1)
            
            return {
                'clean_latent_indices_pre': splits[0],
                'blank_indices': splits[1],
                'latent_indices': splits[2],
                'clean_latent_indices_post': splits[3],
                'clean_latent_indices': torch.cat([splits[0], splits[3]], dim=1)
            }

    def soft_append_pixels(self, history: torch.Tensor, current: torch.Tensor, 
                          overlap: int = 0) -> torch.Tensor:
        """Soft blending of overlapping frames (adapted from FramePack)"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
        
        weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device)
        weights = weights.view(1, 1, -1, 1, 1)
        
        blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)
        
        return output.to(history)


class QuantizedModelLoader:
    """Handles loading and management of quantized models"""
    
    @staticmethod
    def load_quantized_model(model_path: str, model_type: str = 'wan', 
                           quantization: str = 'int8', device: str = 'cuda') -> torch.nn.Module:
        """Load quantized model from checkpoint"""
        
        if quantization == 'fp16':
            model_path = 'downloads/Wan2_1-I2V-ATI-14B_fp16.safetensors'
            state_dict = sf.load_file(model_path, device='cpu')
    
            # Create model (adjust config for 14B if needed)
            model = WanModel(
                model_type='i2v',
                num_layers=32,  # Adjust for 14B model
                dim=5120,        # Match checkpoint dimension
                ffn_dim=13824,
                in_dim=36 
            )
          
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()
            
            # Keep as FP8 or fallback to FP16
            # try:
            #     model = model.to(dtype=torch.float8_e4m3fn, device=device)
            # except:
            #     model = model.to(dtype=torch.float16, device=device)
            model = model.to(dtype=torch.float16, device=device)
            print("Loaded FP16 model successfully")
            
            return model.eval().requires_grad_(False)
        
        elif quantization == 'fp16':
            torch.cuda.empty_cache()
            if model_type == 'wan':
                model = WanModel.from_pretrained(os.path.dirname(model_path))
            model = model.half()
            
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization}")
            
        return model.to(device).eval()


class WanI2VFramePack:
    """WAN Image-to-Video model with FramePack sampling integration"""
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        use_quantization: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.use_quantization = use_quantization
        self.quantized_model_path = quantized_model_path

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # Initialize components
        shard_fn = partial(shard_model, device_id=device_id)
        
        self.text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl")
        self.tokenizer= AutoTokenizer.from_pretrained("google/umt5-xxl")
        
        # # Text encoder
        # self.text_encoder = T5Encoder(
        #     vocab=32128,
        #     dim=4096,
        #     dim_attn=64,
        #     dim_ffn=2048,
        #     num_heads=8,
        #     num_layers=6,
        #     num_buckets=32,
        #     shared_pos=True,
        #     dropout=0.1
        # )
        
        # self.tokenizer = HuggingfaceTokenizer(
        #     name="t5-small",
        #     seq_len=64,
        #     clean="canonicalize"
        # )
        
        # VAE
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 
            subfolder='vae'
        )

        # CLIP
        self.clip_processor = CLIPImageProcessor.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_processor')
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        # self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # Load main model (potentially quantized)
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = QuantizedModelLoader.load_quantized_model(
            checkpoint_dir, 
            model_type='wan',
            quantization='fp8',
            device=self.device
        )

        # Setup USP if needed
        if use_usp:
            from diffusers_helper.wan_components.wan.distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
            
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu and not use_quantization:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        
        # Initialize FramePack sampler
        self.framepack_sampler = FramePackSampler(
            latent_window_size=9,  # Default FramePack window size
            vae_stride=self.vae_stride
        )

    def ensure_device_consistency(self, *tensors, target_device=None):
        """Ensure all tensors are on the same device"""
        if target_device is None:
            target_device = self.device
            
        result = []
        for tensor in tensors:
            if tensor is not None:
                if isinstance(tensor, (list, tuple)):
                    tensor = [t.to(target_device) if hasattr(t, 'to') else t for t in tensor]
                elif hasattr(tensor, 'to'):
                    tensor = tensor.to(target_device)
            result.append(tensor)
        
        return result if len(result) > 1 else result[0]

    def safe_progress_callback(self, progress_callback, section_idx, total_sections, 
                              total_generated_latent_frames, step_idx=None, total_steps=None, 
                              status=None):
        """Safe progress callback that ensures all expected keys are present"""
        if progress_callback is None:
            return
            
        generated_frames = max(0, total_generated_latent_frames * 4 - 3)
        total_seconds = max(0, generated_frames / 30)
        
        info = {
            'section': section_idx + 1,
            'total_sections': total_sections,
            'generated_frames': generated_frames,
            'total_seconds': total_seconds,
            'status': status or f'Processing section {section_idx + 1}/{total_sections}'
        }
        
        if step_idx is not None:
            info['step'] = step_idx + 1
        if total_steps is not None:
            info['total_steps'] = total_steps
            
        progress_callback(info)

    def prepare_wan_noise_inputs(self, noise_latents: torch.Tensor, 
                                clean_latents: torch.Tensor = None) -> List[torch.Tensor]:
        if noise_latents.device != self.device:
            noise_latents = noise_latents.to(self.device)

        if noise_latents.dim() == 5:  # (B, T, C, H, W) or (B, C, T, H, W)
            if noise_latents.shape[1] < 5:  # Probably (B, T, C, H, W)
                noise_latents = noise_latents.permute(0, 2, 1, 3, 4)  # → (B, C, T, H, W)

            # Sanity check print
            print(f"[INFO] After permute: noise_latents shape = {noise_latents.shape}")

            # Split batch
            noise_list = [noise_latents[i] for i in range(noise_latents.shape[0])]
        else:
            raise ValueError("Expected 5D input tensor [B, T, C, H, W] or [B, C, T, H, W]")

        # Each tensor in noise_list should now be [C, T, H, W]
        for i, n in enumerate(noise_list):
            print(f"[CHECK] noise_list[{i}].shape = {n.shape}")

        return noise_list

   

    def prepare_wan_conditional_inputs(
    self, 
    start_latent: torch.Tensor, 
    # clean_latents: torch.Tensor = None, 
    reference_noise: List[torch.Tensor] = None
) -> List[torch.Tensor]:
        """
        Simplified approach: Create conditioning tensors that match reference_noise exactly
        """
        import torch.nn.functional as F
        if reference_noise is None:
            raise ValueError("reference_noise is required to determine target shape")
        
        # Extract the conditioning frame
        if start_latent.dim() == 5:  # (B, C, T, H, W)
            if start_latent.shape[2] == 1:
                conditioning_frame = start_latent.squeeze(2)  # (B, C, H, W)
            else:
                conditioning_frame = start_latent[:, :, 0:1, :, :].squeeze(2)  # Take first frame
        elif start_latent.dim() == 4:  # (B, C, H, W)
            conditioning_frame = start_latent
        elif start_latent.dim() == 3:  # (C, H, W)
            conditioning_frame = start_latent.unsqueeze(0)  # (1, C, H, W)
        else:
            raise ValueError(f"Unexpected start_latent shape: {start_latent.shape}")
        
        print(f"[DEBUG] conditioning_frame shape: {conditioning_frame.shape}")
        
        y_list = []
        for i, ref_tensor in enumerate(reference_noise):
            print(f"[DEBUG] Creating conditioning for ref_tensor[{i}] with shape: {ref_tensor.shape}")
            
            # ref_tensor shape: [C, T, H, W]
            C, T, H, W = ref_tensor.shape
            
            # Extract single conditioning frame (should be first item in batch)
            if conditioning_frame.shape[0] > 1:
                single_frame = conditioning_frame[i]  # (C, H, W)
            else:
                single_frame = conditioning_frame[0]  # (C, H, W)
            
            print(f"[DEBUG] single_frame shape: {single_frame.shape}")
            
            # Resize conditioning frame to match reference spatial dimensions
            if single_frame.shape[-2:] != (H, W):
                print(f"[DEBUG] Resizing conditioning frame from {single_frame.shape[-2:]} to ({H}, {W})")
                single_frame = F.interpolate(
                    single_frame.unsqueeze(0),  # (1, C, H, W)
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # (C, H, W)
            
            # Create conditioning tensor - Option 1: Repeat first frame
            y_tensor = single_frame.unsqueeze(1).repeat(1, T, 1, 1)  # (C, T, H, W)
            
            # Alternative Option 2: First frame only, rest zeros
            # y_tensor = torch.zeros(C, T, H, W, dtype=single_frame.dtype, device=single_frame.device)
            # y_tensor[:, 0] = single_frame
            
            print(f"[DEBUG] Created y_tensor[{i}] with shape: {y_tensor.shape}")
            y_list.append(y_tensor)
        
        return y_list

    

    def wan_sampling_step(self, model_inputs: Dict, timestep: torch.Tensor,
                         guidance_scale: float = 10.0) -> torch.Tensor:
        """Debug version to find where 52 channels come from"""
    
        """Fixed: x=20ch, y=16ch, total=36ch"""
    
        x = model_inputs['noise_latents']  
        context = model_inputs['text_context']
        clip_fea = model_inputs['clip_context'] 
        y = model_inputs.get('conditional_latents', None)
        
        print(clip_fea.shape, 'here is clip shape')
        
        x_input = []
        y_input = []
        
        for x_tensor, y_tensor in zip(x, y):
            # x: 16 channels noise + 4 channels padding = 20 channels
            x_16 = x_tensor[:16]
            padding = torch.zeros(4, *x_16.shape[1:], dtype=x_16.dtype, device=x_16.device)
            x_20 = torch.cat([x_16, padding], dim=0)  # 20 channels
            
            # y: 16 channels conditioning (separate)
            y_16 = y_tensor[:16]  # 16 channels
            
            x_input.append(x_20)
            y_input.append(y_16)
        
        seq_len = int(x_input[0].shape[1] * x_input[0].shape[2] * x_input[0].shape[3] * 1.5)
        
        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            model_output = self.model(
                x=x_input,      # 20 channels
                t=timestep,
                context=context,
                seq_len=seq_len,
                clip_fea=clip_fea,
                y=y_input       # 16 channels
            )
        
        return model_output

    def generate_framepack(
    self,
    input_prompt: str,
    img: Image.Image,
    total_second_length: float = 5.0,
    max_area: int = 720 * 1280,
    shift: float = 5.0,
    sample_solver: str = 'unipc',
    sampling_steps: int = 25,
    guide_scale: float = 10.0,
    n_prompt: str = "",
    seed: int = -1,
    offload_model: bool = True,
    use_teacache: bool = True,
    progress_callback: Optional[callable] = None,
) -> torch.Tensor:
        """
        Generate long video using FramePack's progressive sampling technique integrated with WAN
        """
        print('Starting FramePack generation...')
        
        # Preprocess image
        img_inputs = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
        # Calculate dimensions
        h, w = img_inputs.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // 
            self.patch_size[1] * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // 
            self.patch_size[2] * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        
        # Calculate sections using FramePack strategy
        total_sections, latent_paddings = self.framepack_sampler.calculate_sections(
            total_second_length, fps=30
        )
        
        # Initialize history tensors for progressive generation
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, lat_h, lat_w), 
            dtype=torch.float16,
            device=self.device
        )
        history_pixels = None
        total_generated_latent_frames = 0
        
        # Setup random generator
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        # Encode text prompts
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
            
        if not self.t5_cpu:
            inputs = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)
            n_inputs = self.tokenizer(n_prompt, return_tensors="pt", padding=True, truncation=True)

            self.text_encoder.to(self.device)

            with torch.no_grad():
                context = self.text_encoder(**inputs.to(self.device)).last_hidden_state
                context_null = self.text_encoder(**n_inputs.to(self.device)).last_hidden_state

            if offload_model:
                self.text_encoder.cpu()

        else:
            inputs = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)
            n_inputs = self.tokenizer(n_prompt, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                context = self.text_encoder(**inputs).last_hidden_state
                context_null = self.text_encoder(**n_inputs).last_hidden_state

            # Move outputs to device
            context = context.to(self.device)
            context_null = context_null.to(self.device)
        
        # Encode CLIP features
        # img_clip = (img + 1.0) / 2.0 
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        pixel_values = inputs['pixel_values']  # shape: (1, 3, 224, 224)
        self.clip_model.to(self.device)
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.clip_model(pixel_values=inputs['pixel_values'])
        clip_features = outputs.image_embeds
        expected_dim = 1280
        if clip_features.shape[-1] != expected_dim:
            print(f"[WARNING] CLIP features have {clip_features.shape[-1]} dims, expected {expected_dim}")
            # Create a simple projection layer if needed
            if not hasattr(self, 'clip_projection'):
                self.clip_projection = torch.nn.Linear(
                    clip_features.shape[-1], 
                    expected_dim, 
                    device=self.device, 
                    dtype=torch.float16
                )
            clip_features = self.clip_projection(clip_features)
            if clip_features.dim() == 2:  # [batch, 1280]
                clip_features = clip_features.unsqueeze(1)
        # clip_features = self.clip_model(**inputs).last_hidden_state
        clip_context = clip_features
        
        # Encode first frame with VAE
        if offload_model:
            self.vae.to(self.device)
        img_input = img_inputs.unsqueeze(0).unsqueeze(2)  # (C,H,W) -> (1,C,1,H,W)
    
        # if self.config.offload_models:
        #     self.vae.to(self.device)
        
        try:
            with torch.no_grad():
                latent_dist = self.vae.encode(img_input)
                start_latent = latent_dist.latent_dist.sample()  # (1, C, 1, H, W)
                
                
                
                
            # Remove temporal dimension for conditioning preparation
              # (1, C, H, W)
            
        except torch.cuda.OutOfMemoryError:
            # Fallback: move to CPU
            torch.cuda.empty_cache()
            img_cpu = img_input.cpu()
            self.vae.cpu()
            
            with torch.no_grad():
                latent_dist = self.vae.encode(img_cpu)
                start_latent = latent_dist.latent_dist.sample().to(self.device)
                
            # start_latent = start_latent.squeeze(2)  # (1, C, H, W)
            
            if self.config.offload_models:
                self.vae.to(self.device)
                
        print(f"[DEBUG] Actual start_latent shape: {start_latent.shape}")
        clean_latents_pre = start_latent.to(device=self.device, dtype=history_latents.dtype)

        # Initialize sampling scheduler (adapted for WAN)
        if sample_solver == 'unipc':
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=shift
            )
        else:
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=shift
            )
        
        scheduler.set_timesteps(sampling_steps, device=self.device)
        timesteps = scheduler.timesteps

        # Progressive generation loop (FramePack style)
        for section_idx, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            
            # Safe progress callback
            self.safe_progress_callback(
                progress_callback, section_idx, total_sections, 
                total_generated_latent_frames, 
                status=f'Generating section {section_idx + 1}/{total_sections}'
            )
            
            # Prepare indices for this section
            indices = self.framepack_sampler.prepare_indices(latent_padding)
            
            # Prepare clean latents with proper resizing
            target_shape = start_latent.shape[-2:]  # (H, W)
            
            # Ensure start_latent is on the correct device and dtype
            
            
            
            
            # Ensure history has enough frames
            min_required_frames = 1 + 2 + 16  # 19 frames total
            if history_latents.shape[2] < min_required_frames:
                padding_needed = min_required_frames - history_latents.shape[2]
                padding = torch.zeros(
                    1, 16, padding_needed, *target_shape,
                    dtype=history_latents.dtype,
                    device=self.device
                )
                history_latents = torch.cat([padding, history_latents], dim=2)

            # Extract and resize clean latents
            clean_latents_post, clean_latents_2x, clean_latents_4x = \
                history_latents[:, :, :min_required_frames, :, :].split([1, 2, 16], dim=2)

            def resize_if_needed(tensor, target_shape):
                if tensor.shape[-2:] != target_shape:
                    orig_shape = tensor.shape
                    tensor_flat = tensor.flatten(0, 2)
                    tensor_resized = torch.nn.functional.interpolate(
                        tensor_flat.unsqueeze(1),
                        size=target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    return tensor_resized.unflatten(0, orig_shape[:-2])
                return tensor

            clean_latents_post = resize_if_needed(clean_latents_post, target_shape)
            clean_latents_2x = resize_if_needed(clean_latents_2x, target_shape)
            clean_latents_4x = resize_if_needed(clean_latents_4x, target_shape)
            
            # Ensure all clean latents are on the same device
            clean_latents_pre = clean_latents_pre.to(self.device)
            clean_latents_post = clean_latents_post.to(self.device)
            print('here',clean_latents_pre.shape,clean_latents_post.shape  )
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Generate noise for this section
            num_frames = self.framepack_sampler.latent_window_size * 4 - 3
            noise = torch.randn(
        1, 16, self.framepack_sampler.latent_window_size,  # e.g., (1, 16, 9, H, W)
        lat_h, lat_w,
        dtype=torch.float16,
        generator=seed_g,
        device=self.device
    )
            
            # Setup model
            if offload_model:
                torch.cuda.empty_cache()
            self.model.to(self.device)
            
            # Initialize TeaCache if enabled and supported
            if use_teacache and hasattr(self.model, 'initialize_teacache'):
                self.model.initialize_teacache(enable_teacache=True, num_steps=sampling_steps)
                
            start_latent = start_latent.squeeze(2)
            
            # Prepare inputs for WAN model
            # Convert noise to WAN's expected list format
            noise_list = self.prepare_wan_noise_inputs(noise)
            
            # Conditional latents (conditioning frame) - FIXED: removed the extra comma
            start_latent, clean_latents = self.ensure_device_consistency(start_latent, clean_latents)
            y_list = self.prepare_wan_conditional_inputs(
        start_latent,  # (1, C, H, W)
        
        reference_noise=noise_list
    ) 
        print("Corrected shapes:")
        for i, (x_tensor, y_tensor) in enumerate(zip(noise_list, y_list)):
            print(f"  x[{i}].shape = {x_tensor.shape}, y[{i}].shape = {y_tensor.shape}")
            
            # Denoising loop with FramePack context
            latents = noise
            for step_idx, t in enumerate(timesteps):
                self.safe_progress_callback(
                    progress_callback, section_idx, total_sections,
                    total_generated_latent_frames, step_idx, len(timesteps),
                    status=f'Denoising step {step_idx + 1}/{len(timesteps)}'
                )
                
                # Prepare model inputs
                model_inputs = {
                    'noise_latents': self.prepare_wan_noise_inputs(latents),
                    'text_context': context,
                    'clip_context': clip_context,
                    'conditional_latents': y_list
                }
                
                # Debug prints
                x = model_inputs['noise_latents']
                y = model_inputs['conditional_latents']
                print("WAN model input shapes:")
                for i, (u, v) in enumerate(zip(x, y)):
                    print(f"  x[{i}].shape = {u.shape}, y[{i}].shape = {v.shape}")
                
                # Classifier-free guidance
                if guide_scale > 1.0:
                    # Conditional prediction
                    model_inputs['text_context'] = context
                    noise_pred_cond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                    
                    # Unconditional prediction
                    model_inputs['text_context'] = context_null
                    noise_pred_uncond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                    
                    # Apply guidance
                    noise_pred = []
                    for cond, uncond in zip(noise_pred_cond, noise_pred_uncond):
                        guided = uncond + guide_scale * (cond - uncond)
                        noise_pred.append(guided)
                else:
                    noise_pred = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                
                # Scheduler step
                # Convert back to batch format for scheduler
                if isinstance(noise_pred, list):
                    noise_pred_batch = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred_batch = noise_pred
                    
                latents = scheduler.step(noise_pred_batch, t, latents).prev_sample
            
            # Post-process generated latents
            generated_latents = latents.to(self.device)
            
            if is_last_section:
                start_latent_device = start_latent.to(self.device)
                generated_latents = torch.cat([start_latent_device, generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            
            # Move generated_latents to same device as history_latents before concatenation
            generated_latents_for_history = generated_latents.to(history_latents.device, dtype=history_latents.dtype)
            history_latents = torch.cat([generated_latents_for_history, history_latents], dim=2)

            # Decode to pixels
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
                self.vae.to(self.device)

            # Move history_latents to GPU for VAE decoding
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :].to(self.device)

            if history_pixels is None:
                with torch.no_grad():
                    history_pixels = self.vae.decode(real_history_latents).sample.cpu()
            else:
                section_latent_frames = (self.framepack_sampler.latent_window_size * 2 + 1) if is_last_section else (self.framepack_sampler.latent_window_size * 2)
                overlapped_frames = self.framepack_sampler.latent_window_size * 4 - 3

                with torch.no_grad():
                    current_pixels = self.vae.decode(real_history_latents[:, :, :section_latent_frames]).sample.cpu()
                    history_pixels = self.framepack_sampler.soft_append_pixels(
                        current_pixels, history_pixels, overlapped_frames
                    )

            if offload_model:
                self.vae.cpu()
                torch.cuda.empty_cache()

            if is_last_section:
                break
        
        # Clean up
        if offload_model:
            self.model.cpu()
            self.vae.cpu()
            torch.cuda.empty_cache()
        
        # Final processing callback
        self.safe_progress_callback(
            progress_callback, total_sections - 1, total_sections,
            total_generated_latent_frames,
            status='Generation complete!'
        )
        
        return history_pixels

    # def generate_standard(
    #     self,
    #     input_prompt: str,
    #     img: Image.Image,
    #     num_frames: int = 33,  # Standard WAN generation length
    #     max_area: int = 720 * 1280,
    #     shift: float = 5.0,
    #     sample_solver: str = 'unipc',
    #     sampling_steps: int = 25,
    #     guide_scale: float = 10.0,
    #     n_prompt: str = "",
    #     seed: int = -1,
    #     offload_model: bool = True,
    # ) -> torch.Tensor:
    #     """Standard WAN generation for shorter videos"""
        
    #     # Preprocess image
    #     img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
    #     # Calculate dimensions
    #     h, w = img.shape[1:]
    #     aspect_ratio = h / w
    #     lat_h = round(
    #         np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // 
    #         self.patch_size[1] * self.patch_size[1]
    #     )
    #     lat_w = round(
    #         np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // 
    #         self.patch_size[2] * self.patch_size[2]
    #     )
    #     h = lat_h * self.vae_stride[1]
    #     w = lat_w * self.vae_stride[2]
        
    #     # Setup random generator
    #     seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    #     seed_g = torch.Generator(device=self.device)
    #     seed_g.manual_seed(seed)
        
    #     # Text encoding
    #     if n_prompt == "":
    #         n_prompt = self.sample_neg_prompt
            
    #     input_ids, attention_mask = self.tokenizer(input_prompt, return_mask=True)
    #     ninput_ids, nattention_mask = self.tokenizer(n_prompt, return_mask=True)
        
    #     if not self.t5_cpu:
    #         self.text_encoder.model.to(self.device)
    #         context = self.text_encoder(input_ids, attention_mask)
    #         context_null = self.text_encoder(ninput_ids, nattention_mask)
    #         if offload_model:
    #             self.text_encoder.model.cpu()
    #     else:
    #         context = self.text_encoder(input_ids, attention_mask)
    #         context_null = self.text_encoder(ninput_ids, nattention_mask)
    #         context = [t.to(self.device) for t in context]
    #         context_null = [t.to(self.device) for t in context_null]
        
    #     # CLIP encoding
    #     img_clip = (img + 1.0) / 2.0 
    #     inputs = self.clip_processor(images=img_clip, return_tensors="pt").to(self.device)
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     clip_features = self.clip_model(**inputs).last_hidden_state
        
    #     # VAE encoding
    #     if offload_model:
    #         self.vae.to(self.device)
    #     img_input = img.unsqueeze(0).unsqueeze(2)
        
    #     with torch.no_grad():
    #         latent_dist = self.vae.encode(img_input)
    #         start_latent = latent_dist.latent_dist.sample()
        
    #     # Generate noise
    #     noise = torch.randn(
    #         1, 16, num_frames,
    #         lat_h, lat_w,
    #         dtype=torch.float32,
    #         generator=seed_g,
    #         device=self.device
    #     )
        
    #     # Setup scheduler
    #     if sample_solver == 'unipc':
    #         scheduler = FlowUniPCMultistepScheduler(
    #             num_train_timesteps=self.num_train_timesteps,
    #             shift=shift
    #         )
    #     else:
    #         scheduler = FlowDPMSolverMultistepScheduler(
    #             num_train_timesteps=self.num_train_timesteps,
    #             shift=shift
    #         )
        
    #     scheduler.set_timesteps(sampling_steps, device=self.device)
    #     timesteps = scheduler.timesteps
        
    #     # Prepare model inputs
    #     noise_list = self.prepare_wan_noise_inputs(noise)
    #     y_list = self.prepare_wan_conditional_inputs(start_latent,reference_noise=self.prepare_wan_noise_inputs(latents))
        
    #     # Setup model
    #     if offload_model:
    #         torch.cuda.empty_cache()
    #     self.model.to(self.device)
        
    #     # Denoising loop
    #     latents = noise
    #     for step_idx, t in enumerate(timesteps):
    #         model_inputs = {
    #             'noise_latents': self.prepare_wan_noise_inputs(latents),
    #             'text_context': context,
    #             'clip_context': clip_features,
    #             'conditional_latents': y_list
    #         }
            
    #         # Classifier-free guidance
    #         if guide_scale > 1.0:
    #             model_inputs['text_context'] = context
    #             noise_pred_cond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                
    #             model_inputs['text_context'] = context_null
    #             noise_pred_uncond = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
                
    #             noise_pred = []
    #             for cond, uncond in zip(noise_pred_cond, noise_pred_uncond):
    #                 guided = uncond + guide_scale * (cond - uncond)
    #                 noise_pred.append(guided)
    #         else:
    #             noise_pred = self.wan_sampling_step(model_inputs, t.unsqueeze(0))
            
    #         # Scheduler step
    #         if isinstance(noise_pred, list):
    #             noise_pred_batch = torch.stack(noise_pred, dim=0)
    #         else:
    #             noise_pred_batch = noise_pred
                
    #         latents = scheduler.step(noise_pred_batch, t, latents).prev_sample
        
    #     # Decode to pixels
    #     if offload_model:
    #         self.model.cpu()
    #         torch.cuda.empty_cache()
    #         self.vae.to(self.device)
        
    #     # Concatenate start frame with generated frames
    #     final_latents = torch.cat([start_latent, latents], dim=2)
        
    #     with torch.no_grad():
    #         pixels = self.vae.decode(final_latents).sample
        
    #     if offload_model:
    #         self.vae.cpu()
    #         torch.cuda.empty_cache()
        
    #     return pixels

    def generate(
        self,
        input_prompt: str,
        img: Image.Image,
        total_second_length: float = 5.0,
        use_framepack: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Main generation method with FramePack integration
        
        Args:
            use_framepack: Whether to use FramePack's progressive sampling (recommended for long videos)
        """
        if use_framepack and total_second_length > 3.0:
            # Use FramePack for videos longer than 3 seconds
            return self.generate_framepack(
                input_prompt=input_prompt,
                img=img,
                total_second_length=total_second_length,
                **kwargs
            )
        else:
            # Use standard generation for short videos
            num_frames = int(total_second_length * 30 / 4)  # Convert to latent frames
            num_frames = max(num_frames, 9)  # Minimum frames
            return self.generate_standard(
                input_prompt=input_prompt,
                img=img,
                num_frames=num_frames,
                **kwargs
            )

    @torch.no_grad()
    def encode_video_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Encode a list of video frames to latents"""
        # Convert PIL images to tensor
        frame_tensors = []
        for frame in frames:
            frame_tensor = TF.to_tensor(frame).sub_(0.5).div_(0.5)
            frame_tensors.append(frame_tensor)
        
        # Stack into video tensor (C, T, H, W)
        video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).to(self.device)
        
        # Encode with VAE
        self.vae.to(self.device)
        latent_dist = self.vae.encode(video_tensor)
        latents = latent_dist.latent_dist.sample()
        
        return latents

    @torch.no_grad()
    def decode_latents_to_video(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode latents back to video frames"""
        self.vae.to(self.device)
        
        # Decode latents
        pixels = self.vae.decode(latents).sample
        
        # Convert to PIL images
        pixels = pixels.squeeze(0)  # Remove batch dimension
        pixels = (pixels + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        pixels = pixels.clamp(0, 1)
        
        frames = []
        for t in range(pixels.shape[1]):
            frame_tensor = pixels[:, t]  # (C, H, W)
            frame_pil = TF.to_pil_image(frame_tensor)
            frames.append(frame_pil)
        
        return frames

    def save_video_mp4(self, pixels: torch.Tensor, output_path: str, fps: int = 30, crf: int = 16):
        """Save generated video as MP4 file"""
        import cv2
        import tempfile
        import os
        
        # Convert tensor to numpy
        pixels = pixels.squeeze(0)  # Remove batch dimension
        pixels = (pixels + 1.0) / 2.0  # Denormalize
        pixels = (pixels * 255).clamp(0, 255).byte()
        pixels = pixels.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        
        # Setup video writer
        height, width = pixels.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in pixels:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        # Re-encode with better compression if needed
        if crf < 16:
            temp_path = output_path.replace('.mp4', '_temp.mp4')
            os.rename(output_path, temp_path)
            
            cmd = f'ffmpeg -i {temp_path} -c:v libx264 -crf {crf} -pix_fmt yuv420p {output_path}'
            os.system(cmd)
            os.remove(temp_path)

    def interpolate_frames(self, start_frame: Image.Image, end_frame: Image.Image, 
                          num_interpolations: int = 8) -> List[Image.Image]:
        """Generate interpolated frames between two images using WAN"""
        # This could be extended to use WAN for frame interpolation
        # For now, return linear interpolation as placeholder
        
        start_tensor = TF.to_tensor(start_frame)
        end_tensor = TF.to_tensor(end_frame)
        
        interpolated = []
        for i in range(num_interpolations + 2):
            alpha = i / (num_interpolations + 1)
            interp_tensor = (1 - alpha) * start_tensor + alpha * end_tensor
            interp_frame = TF.to_pil_image(interp_tensor)
            interpolated.append(interp_frame)
        
        return interpolated


# Example usage and testing functions
class WanFramePackDemo:
    """Demo class showing how to use WAN with FramePack integration"""
    
    def __init__(self, checkpoint_dir: str, device_id: int = 0):
        self.config = WanFramePackConfig()
        self.model = WanI2VFramePack(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            use_quantization='fp8',
            # offload_model=True
        )
    
    def generate_short_video(self, image_path: str, prompt: str, duration: float = 3.0) -> str:
        """Generate a short video (standard WAN generation)"""
        img = Image.open(image_path).convert('RGB')
        
        def progress_callback(info):
            if 'step' in info:
                print(f"Step {info['step']}/{info['total_steps']}")
            else:
                print(f"Section {info['section']}/{info['total_sections']}")
        
        pixels = self.model.generate(
            input_prompt=prompt,
            img=img,
            total_second_length=duration,
            use_framepack=False,  # Use standard generation
            sampling_steps=25,
            guide_scale=10.0,
            progress_callback=progress_callback
        )
        
        # Save video
        output_path = f"short_video_{duration}s.mp4"
        self.model.save_video_mp4(pixels, output_path, fps=30)
        return output_path
    
    def generate_long_video(self, image_path: str, prompt: str, duration: float = 60.0) -> str:
        """Generate a long video using FramePack"""
        img = Image.open(image_path).convert('RGB')
        
        def progress_callback(info):
            if 'status' in info:
                print(info['status'])
            if 'section' in info and 'total_sections' in info:
                section_progress = (info['section'] / info['total_sections']) * 100
                print(f"Overall progress: {section_progress:.1f}% - "
                      f"Generated {info.get('generated_frames', 0)} frames "
                      f"({info.get('total_seconds', 0):.1f}s)")
        
        pixels = self.model.generate_framepack(
            input_prompt=prompt,
            img=img,
            total_second_length=duration,
            sampling_steps=25,
            guide_scale=10.0,
            use_teacache=True,
            progress_callback=progress_callback
        )
        
        # Save video
        output_path = f"long_video_{duration}s.mp4"
        self.model.save_video_mp4(pixels, output_path, fps=30)
        return output_path
    
    def benchmark_framepack_vs_standard(self, image_path: str, prompt: str):
        """Compare FramePack vs standard generation for same duration"""
        import time
        
        img = Image.open(image_path).convert('RGB')
        test_duration = 10.0  # 10 seconds
        
        print("Testing Standard Generation...")
        start_time = time.time()
        pixels_standard = self.model.generate(
            input_prompt=prompt,
            img=img,
            total_second_length=test_duration,
            use_framepack=False,
            sampling_steps=25
        )
        standard_time = time.time() - start_time
        
        print("Testing FramePack Generation...")
        start_time = time.time()
        pixels_framepack = self.model.generate_framepack(
            input_prompt=prompt,
            img=img,
            total_second_length=test_duration,
            sampling_steps=25,
            use_teacache=True
        )
        framepack_time = time.time() - start_time
        
        print(f"\nResults:")
        print(f"Standard generation: {standard_time:.2f}s")
        print(f"FramePack generation: {framepack_time:.2f}s")
        print(f"FramePack speedup: {standard_time/framepack_time:.2f}x")
        
        # Save both videos for comparison
        self.model.save_video_mp4(pixels_standard, "standard_10s.mp4")
        self.model.save_video_mp4(pixels_framepack, "framepack_10s.mp4")
        
        return {
            'standard_time': standard_time,
            'framepack_time': framepack_time,
            'speedup': standard_time / framepack_time
        }


def test_wan_framepack_integration():
    """Test function to verify the integration works correctly"""
    
    # Example configuration
    checkpoint_dir ="downloads/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
    
    try:
        # Initialize demo
        demo = WanFramePackDemo(checkpoint_dir)
        
        # Test short video generation
        # print("Testing short video generation...")
        # short_video = demo.generate_short_video(
        #     image_path="random-image.jpg",
        #     prompt="A boy dancing gracefully with clear movements",
        #     duration=3.0
        # )
        # print(f"Generated short video: {short_video}")
        
        # Test long video generation with FramePack
        print("\nTesting long video generation with FramePack...")
        long_video = demo.generate_long_video(
            image_path="random-image.jpg", 
            prompt="A boy dancing gracefully with clear movements",
            duration=30.0
        )
        print(f"Generated long video: {long_video}")
        
        # Benchmark comparison
        # print("\nRunning benchmark...")
        # results = demo.benchmark_framepack_vs_standard(
        #     image_path="random-image.jpg",
        #     prompt="A boy dancing gracefully with clear movements"
        # )
        # print(f"Benchmark results: {results}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    test_wan_framepack_integration()
