"""
Utility functions for Mural Restoration Project
Common functions for image loading, metric calculations, and data processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch


def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        grayscale: If True, load as grayscale
        
    Returns:
        Image as numpy array
    """
    if grayscale:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, save_path: str):
    """
    Save an image to file path.
    
    Args:
        image: Image as numpy array (RGB format)
        save_path: Path to save image
    """
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    cv2.imwrite(str(save_path), image_bgr)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Calculate PSNR between two images.
    
    Args:
        img1: First image
        img2: Second image
        data_range: Data range (default: 255 for uint8)
        
    Returns:
        PSNR value
    """
    if data_range is None:
        data_range = 255.0 if img1.dtype == np.uint8 else 1.0
    return psnr(img1, img2, data_range=data_range)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, multichannel: bool = True) -> float:
    """
    Calculate SSIM between two images.
    
    Args:
        img1: First image
        img2: Second image
        multichannel: If True, treat as multichannel image
        
    Returns:
        SSIM value
    """
    if len(img1.shape) == 3:
        return ssim(img1, img2, multichannel=multichannel, channel_axis=2, data_range=255)
    else:
        return ssim(img1, img2, data_range=255)


def calculate_lpips(img1: np.ndarray, img2: np.ndarray, lpips_model=None) -> float:
    """
    Calculate LPIPS (perceptual similarity) between two images.
    
    Args:
        img1: First image (RGB, uint8)
        img2: Second image (RGB, uint8)
        lpips_model: Pre-loaded LPIPS model (will create if None)
        
    Returns:
        LPIPS value
    """
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex')
    
    # Convert to tensor and normalize
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    img1_tensor = img1_tensor.unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0)
    
    # Calculate LPIPS
    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
    
    return dist.item()


def calculate_edge_accuracy(img1: np.ndarray, img2: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> float:
    """
    Calculate edge accuracy using Canny edge detection.
    
    Args:
        img1: First image
        img2: Second image
        threshold1: Canny lower threshold
        threshold2: Canny upper threshold
        
    Returns:
        Edge accuracy (intersection over union of edges)
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2
    
    # Detect edges
    edges1 = cv2.Canny(gray1, threshold1, threshold2)
    edges2 = cv2.Canny(gray2, threshold1, threshold2)
    
    # Calculate intersection and union
    intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
    union = np.logical_or(edges1 > 0, edges2 > 0).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_image_files(directory: str, pattern: str = "*.png") -> List[Path]:
    """
    Get all image files from directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (default: *.png)
        
    Returns:
        List of image file paths
    """
    return sorted(Path(directory).glob(pattern))


def visualize_comparison(images: List[np.ndarray], titles: List[str], figsize: Tuple[int, int] = (15, 5)):
    """
    Visualize multiple images side by side.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

