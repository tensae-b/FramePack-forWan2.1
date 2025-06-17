#!/usr/bin/env python
"""
Example usage of Wan model with FramePack sampling and quantization
"""

import os
import torch
from PIL import Image
from easydict import EasyDict
import logging
from trial_back import WanI2VFramePack
# Configure logging
logging.basicConfig(level=logging.INFO)

def create_config():
    """Create configuration for Wan model"""
    config = EasyDict()
    
    # Model parameters
    config.num_train_timesteps = 1000
    config.param_dtype = torch.bfloat16
    
    # Text encoder config
    config.text_len = 512
    config.t5_dtype = torch.bfloat16
    config.t5_checkpoint = "t5_xxl.pt"
    config.t5_tokenizer = "t5_tokenizer"
    
    # VAE config
    config.vae_stride = [4, 8, 8]
    config.patch_size = [1, 2, 2]
    config.vae_checkpoint = "vae.pt"
    
    # CLIP config
    config.clip_dtype = torch.float16
    config.clip_checkpoint = "clip.pt"
    config.clip_tokenizer = "clip_tokenizer"
    
    # Sampling config
    config.sample_neg_prompt = "blurry, low quality, distorted, deformed"
    
    return config


def main():
    
    from huggingface_hub import hf_hub_download
    import os
    import shutil
    file_path = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy", 
        filename="Wan2_1-I2V-ATI-14B_fp16.safetensors"
    )
    print(file_path)
    target_dir = "/downloads"
    os.makedirs(target_dir, exist_ok=True)  # Create it if it doesn't exist

    # Move the file
    shutil.copy(file_path, os.path.join(target_dir, "Wan2_1-I2V-ATI-14B_fp16.safetensors"))
    # Setup paths
    checkpoint_dir = "downloads/Wan2_1-I2V-ATI-14B_fp16.safetensors"
    quantized_model_path = "/path/to/quantized/wan_model_int8.pt"  # Optional
    
    # Load image
    input_image_path = "random-image.jpg"
    img = Image.open(input_image_path).convert('RGB')
    
    # Create config
    config = create_config()
    
    # Initialize model with quantization
    model = WanI2VFramePack(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        use_quantization='fp16',  # Options: 'int8', 'fp16', 'bnb', None
        quantized_model_path=quantized_model_path,
        # offload_model=True,  # Important for memory management
        t5_cpu=True,  # Keep T5 on CPU to save GPU memory
    )
    
    # Define progress callback
    def progress_callback(info):
        print(f"Section {info['section']}/{info['total_sections']} - "
              f"Generated {info['generated_frames']} frames "
              f"({info['total_seconds']:.1f} seconds)")
    
    # Generate 1-minute video using FramePack technique
    prompt = "A boy dancing gracefully with clear movements, full of energy and charm"
    
    video_tensor = model.generate_framepack(
        input_prompt=prompt,
        img=img,
        total_second_length=5.0,  # 1 minute
        max_area=480*854,  # Lower resolution for memory efficiency
        shift=3.0,  # Lower shift for 480p as recommended
        sample_solver='unipc',
        sampling_steps=25,  # FramePack default
        guide_scale=10.0,  # FramePack default
        n_prompt="",  # Will use default negative prompt
        seed=42,
        offload_model=True,
        use_teacache=True,  # Enable TeaCache for faster generation
        progress_callback=progress_callback,
    )
    
    # Save the video
    save_video_tensor(video_tensor, "output_60s_video.mp4", fps=30)
    print("Video generation complete!")


def save_video_tensor(tensor, output_path, fps=30, crf=16):
    """Save video tensor to MP4 file"""
    import cv2
    import numpy as np
    
    # Convert tensor to numpy
    # Assuming tensor shape: (C, T, H, W)
    video = tensor.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
    video = ((video + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Setup video writer
    height, width = video.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in video:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {output_path}")


def example_with_memory_monitoring():
    """Example with GPU memory monitoring"""
    import GPUtil
    import psutil
    
    # Monitor initial memory
    print(f"Initial GPU memory: {GPUtil.getGPUs()[0].memoryUsed:.1f} MB")
    print(f"Initial CPU memory: {psutil.virtual_memory().percent:.1f}%")
    
    # Setup
    checkpoint_dir = "/path/to/wan/checkpoints"
    config = create_config()
    
    # Initialize with aggressive memory optimization
    model = WanI2VFramePack(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        use_quantization='bnb',  # Use bitsandbytes for 8-bit quantization
        quantized_model_path=None,  # Will quantize on-the-fly
        offload_model=True,
        t5_cpu=True,
        init_on_cpu=True,  # Initialize on CPU first
    )
    
    print(f"After model init - GPU: {GPUtil.getGPUs()[0].memoryUsed:.1f} MB")
    
    # Load a smaller test image
    img = Image.new('RGB', (512, 512), color='white')
    
    # Generate with memory tracking
    def memory_aware_callback(info):
        gpu = GPUtil.getGPUs()[0]
        print(f"Section {info['section']}/{info['total_sections']} - "
              f"GPU: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB "
              f"({gpu.memoryUtil*100:.1f}%)")
    
    # Generate shorter video first as test
    video_tensor = model.generate_framepack(
        input_prompt="A simple animated scene",
        img=img,
        total_second_length=10.0,  # Start with 10 seconds
        max_area=384*384,  # Very low resolution for testing
        shift=3.0,
        sampling_steps=10,  # Fewer steps for testing
        guide_scale=7.5,
        offload_model=True,
        use_teacache=True,
        progress_callback=memory_aware_callback,
    )
    
    print("Test generation complete!")


def batch_generation_example():
    """Example of generating multiple videos in sequence"""
    checkpoint_dir = "/path/to/wan/checkpoints"
    config = create_config()
    
    # Initialize model once
    model = WanI2VFramePack(
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_quantization='fp16',  # FP16 for balance of speed and quality
        offload_model=True,
    )
    
    # List of prompts and images
    tasks = [
        {
            "image": "person1.jpg",
            "prompt": "Person dancing energetically",
            "duration": 30.0,
        },
        {
            "image": "person2.jpg", 
            "prompt": "Person walking gracefully",
            "duration": 45.0,
        },
        {
            "image": "person3.jpg",
            "prompt": "Person doing yoga poses",
            "duration": 60.0,
        }
    ]
    
    # Process each task
    for i, task in enumerate(tasks):
        print(f"\nProcessing task {i+1}/{len(tasks)}")
        
        img = Image.open(task["image"]).convert('RGB')
        
        video_tensor = model.generate_framepack(
            input_prompt=task["prompt"],
            img=img,
            total_second_length=task["duration"],
            max_area=480*854,
            seed=42 + i,  # Different seed for each
            offload_model=True,
            use_teacache=True,
        )
        
        output_path = f"output_video_{i+1}.mp4"
        save_video_tensor(video_tensor, output_path, fps=30)
        
        # Clear cache between generations
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def advanced_configuration_example():
    """Example with advanced configuration options"""
    
    # Custom configuration
    config = EasyDict()
    
    # Model architecture settings
    config.num_train_timesteps = 1000
    config.param_dtype = torch.float16  # Use FP16 for lower memory
    
    # Adjust model dimensions for efficiency
    config.text_len = 256  # Shorter context for faster processing
    config.t5_dtype = torch.float16
    config.t5_checkpoint = "t5_base.pt"  # Use smaller T5 model
    config.t5_tokenizer = "t5_tokenizer"
    
    # VAE settings
    config.vae_stride = [4, 8, 8]
    config.patch_size = [1, 2, 2]
    config.vae_checkpoint = "vae_fp16.pt"  # FP16 VAE
    
    # CLIP settings
    config.clip_dtype = torch.float16
    config.clip_checkpoint = "clip_fp16.pt"
    config.clip_tokenizer = "clip_tokenizer"
    
    # Quality vs speed trade-offs
    config.sample_neg_prompt = ""  # Empty for faster generation
    
    # Initialize with custom config
    model = WanI2VFramePack(
        config=config,
        checkpoint_dir="/path/to/optimized/checkpoints",
        use_quantization='int8',
        offload_model=True,
        t5_cpu=True,
    )
    
    # Generate with custom settings
    img = Image.open("test_image.jpg").convert('RGB')
    
    # Fast generation settings
    fast_video = model.generate_framepack(
        input_prompt="Quick test animation",
        img=img,
        total_second_length=5.0,
        max_area=320*320,  # Very low res
        sampling_steps=10,  # Few steps
        guide_scale=5.0,  # Lower guidance
        use_teacache=True,
    )
    
    # Quality generation settings
    quality_video = model.generate_framepack(
        input_prompt="High quality cinematic scene",
        img=img,
        total_second_length=10.0,
        max_area=720*1280,  # High res
        sampling_steps=40,  # More steps
        guide_scale=10.0,  # Higher guidance
        use_teacache=False,  # Disable for best quality
    )


def distributed_generation_example():
    """Example for multi-GPU generation"""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    config = create_config()
    
    # Initialize with distributed settings
    model = WanI2VFramePack(
        config=config,
        checkpoint_dir="/path/to/checkpoints",
        device_id=local_rank,
        rank=local_rank,
        t5_fsdp=True,  # Enable FSDP for T5
        dit_fsdp=True,  # Enable FSDP for DiT
        use_quantization='fp16',
        offload_model=False,  # Don't offload with FSDP
    )
    
    # Only process on rank 0
    if local_rank == 0:
        img = Image.open("input.jpg").convert('RGB')
        video_tensor = model.generate_framepack(
            input_prompt="Distributed generation test",
            img=img,
            total_second_length=30.0,
            offload_model=False,
        )
        save_video_tensor(video_tensor, "distributed_output.mp4")
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run other examples:
    # example_with_memory_monitoring()
    # batch_generation_example()
    # advanced_configuration_example()
    # distributed_generation_example()