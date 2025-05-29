import torch
import os
from .wan_components.wan_vae_model import WanVAE # Import the custom WanVAE

# --- Configuration for WanVAE ---
# You'll need to make sure this .pth file is accessible.
# It's best to make this path configurable, e.g., via an environment variable or argument.
DEFAULT_WAN_VAE_PATH = "cache/vae_step_411000.pth" # Default path from the WanVAE code
# Check if an environment variable is set for the VAE path
# You could also pass this path as an argument to load_wan_vae
WAN_VAE_PATH = os.environ.get("WAN_VAE_PTH_PATH", DEFAULT_WAN_VAE_PATH)

def load_wan_vae(
    vae_pth_path: str = WAN_VAE_PATH,
    z_dim: int = 16, # Matches the default in WanVAE and HunyuanVideoPatchEmbed
    dtype: torch.dtype = torch.float16, # FramePack generally uses float16 for VAE
    device: str = "cpu" # Load to CPU initially, like other models in FramePack
):
    """
    Loads the WanVAE model.
    """
    if not os.path.exists(vae_pth_path):
        raise FileNotFoundError(
            f"WanVAE checkpoint not found at {vae_pth_path}. "
            "Please ensure the .pth file is correctly placed or update the path."
        )
    try:
        # Note: The WanVAE class itself handles moving the underlying model to the specified device.
        # The dtype is also handled by WanVAE's internal amp.autocast and tensor conversions.
        # We pass the device, but the initial model loading in _video_vae uses its own device param for torch.load
        # The final .to(device) in WanVAE.__init__ should place it correctly.
        print(f"Loading WanVAE from: {vae_pth_path} with z_dim={z_dim}, target device={device}, target dtype={dtype}")
        
        # The WanVAE class takes 'device' and 'dtype' arguments.
        # The 'device' in _video_vae is for map_location during torch.load.
        # The 'device' in WanVAE.__init__ is for the final .to(device).
        vae_model = WanVAE(
            z_dim=z_dim,
            vae_pth=vae_pth_path, # Path to the .pth file
            dtype=dtype,          # Target dtype for operations
            device=device         # Target device for the model
        )
        print(f"WanVAE loaded. Internal model device: {next(vae_model.model.parameters()).device}")
        return vae_model
    except Exception as e:
        print(f"Error loading WanVAE: {e}")
        raise

def vae_encode_wan(
    image_tensor: torch.Tensor, # Expected shape: B x C x T x H x W or C x T x H x W
    wan_vae_model: WanVAE,
    **kwargs # For API consistency, not used by WanVAE.encode directly
) -> torch.Tensor:
    """
    Encodes an image tensor or a batch of image tensors using the WanVAE model.
    The WanVAE.encode method expects a list of videos [C, T, H, W].
    """
    original_device = image_tensor.device
    # WanVAE expects input on its own device and with its specified dtype
    image_tensor_for_vae = image_tensor.to(device=wan_vae_model.device, dtype=wan_vae_model.dtype)

    if image_tensor_for_vae.ndim == 4: # Single video C x T x H x W
        image_list = [image_tensor_for_vae]
    elif image_tensor_for_vae.ndim == 5: # Batch of videos B x C x T x H x W
        image_list = [img_slice for img_slice in image_tensor_for_vae]
    else:
        raise ValueError(f"Unsupported image_tensor ndim: {image_tensor_for_vae.ndim}. Expected 4 or 5.")

    # The WanVAE.encode method returns a list of latent tensors.
    latents_list = wan_vae_model.encode(image_list)

    # Stack the list of latents back into a single tensor if it was a batch
    if not latents_list:
        raise ValueError("WanVAE.encode returned an empty list.")
    
    # Ensure latents are on the original device
    stacked_latents = torch.stack(latents_list).to(original_device)

    # The WanVAE's encode method already applies its internal scaling.
    # The output latents should be ready for the transformer if dimensions match.
    # z_dim is 16 for WanVAE, which matches HunyuanVideoPatchEmbed's in_channels.
    return stacked_latents

def vae_decode_wan(
    latents_tensor: torch.Tensor, # Expected shape: B x Z_DIM x T_latent x H_latent x W_latent or Z_DIM x T_l x H_l x W_l
    wan_vae_model: WanVAE,
    **kwargs # For API consistency
) -> torch.Tensor:
    """
    Decodes a latent tensor or a batch of latent tensors using the WanVAE model.
    The WanVAE.decode method expects a list of latents.
    """
    original_device = latents_tensor.device
    # WanVAE expects input on its own device and with its specified dtype
    latents_tensor_for_vae = latents_tensor.to(device=wan_vae_model.device, dtype=wan_vae_model.dtype)

    if latents_tensor_for_vae.ndim == 4: # Single latent Z_DIM x T_l x H_l x W_l
        latents_list_for_vae = [latents_tensor_for_vae]
    elif latents_tensor_for_vae.ndim == 5: # Batch of latents B x Z_DIM x T_l x H_l x W_l
        latents_list_for_vae = [latent_slice for latent_slice in latents_tensor_for_vae]
    else:
        raise ValueError(f"Unsupported latents_tensor ndim: {latents_tensor_for_vae.ndim}. Expected 4 or 5.")

    # The WanVAE.decode method returns a list of decoded video tensors.
    decoded_videos_list = wan_vae_model.decode(latents_list_for_vae)

    if not decoded_videos_list:
        raise ValueError("WanVAE.decode returned an empty list.")

    # Stack the list back into a single tensor and ensure it's on the original device
    stacked_decoded_videos = torch.stack(decoded_videos_list).to(original_device)
    
    # WanVAE.decode already clamps output to [-1, 1].
    return stacked_decoded_videos

def vae_decode_fake_wan(
    latents_tensor: torch.Tensor,
    wan_vae_model: WanVAE,
    **kwargs
) -> torch.Tensor:
    """
    Fake decode for previews. WanVAE doesn't have a specific 'fake' decode.
    So, we just call the full decode.
    """
    # print("Warning: vae_decode_fake_wan is calling full vae_decode_wan as WanVAE has no separate fake decode.")
    return vae_decode_wan(latents_tensor, wan_vae_model, **kwargs)