import gradio as gr
import os
import shutil
from pathlib import Path
from PIL import Image, ExifTags, ImageFilter, ImageEnhance
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from datetime import datetime
import piexif
import json
import logging
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter
import random
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import re
import time

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.raw', '.heic', '.heif'
}

# Social media presets
DIMENSION_PRESETS = {
    "Custom": (1920, 1080),
    "Full HD (1920x1080)": (1920, 1080),
    "4K UHD (3840x2160)": (3840, 2160),
    "Instagram Square (1080x1080)": (1080, 1080),
    "Instagram Portrait (1080x1350)": (1080, 1350),
    "Instagram Story (1080x1920)": (1080, 1920),
    "Twitter Post (1200x675)": (1200, 675),
    "Facebook Cover (1200x630)": (1200, 630),
    "YouTube Thumbnail (1280x720)": (1280, 720),
    "Web Banner (1920x600)": (1920, 600),
    "Mobile Screen (414x896)": (414, 896),
    "Tablet Screen (1024x768)": (1024, 768)
}

# Platform-specific optimization presets
PLATFORM_PRESETS = {
    "general": {
        "max_dimensions": (1920, 1080),
        "format": "Auto",
        "quality": 85,
        "description": "General web optimization"
    },
    "instagram": {
        "max_dimensions": (1080, 1080),
        "format": "JPEG",
        "quality": 90,
        "description": "Instagram optimized (square format)"
    },
    "twitter": {
        "max_dimensions": (1200, 675),
        "format": "JPEG", 
        "quality": 85,
        "description": "Twitter post optimization"
    },
    "web_gallery": {
        "max_dimensions": (2048, 2048),
        "format": "Auto",
        "quality": 90,
        "description": "High-quality web gallery"
    },
    "email": {
        "max_dimensions": (800, 600),
        "format": "JPEG",
        "quality": 75,
        "description": "Email-friendly small size"
    },
    "facebook": {
        "max_dimensions": (1200, 630),
        "format": "JPEG",
        "quality": 85,
        "description": "Facebook post optimization"
    },
    "linkedin": {
        "max_dimensions": (1200, 627),
        "format": "JPEG",
        "quality": 85,
        "description": "LinkedIn post optimization"
    },
    "pinterest": {
        "max_dimensions": (1000, 1500),
        "format": "JPEG",
        "quality": 85,
        "description": "Pinterest pin optimization"
    }
}

# Format-specific optimization settings
FORMAT_SETTINGS = {
    "JPEG": {
        "photo_quality": 90,
        "web_quality": 85,
        "progressive": True,
        "optimize": True
    },
    "PNG": {
        "compress_level": 6,
        "optimize": True
    },
    "WEBP": {
        "quality": 90,
        "lossless": False,
        "method": 6,
        "optimize": True
    }
}

def find_images(source_dir: str) -> List[Path]:
    """Recursively find all image files in directory"""
    if not source_dir or not os.path.exists(source_dir):
        return []
    
    source_path = Path(source_dir)
    image_files = []
    
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file_path)
    
    return sorted(image_files)

def create_destination_structure(source_file: Path, source_dir: Path, dest_dir: Path, preserve_structure: bool) -> Path:
    """Create destination path maintaining folder structure if requested"""
    if preserve_structure:
        # Calculate relative path from source directory
        relative_path = source_file.relative_to(source_dir)
        dest_path = dest_dir / relative_path.parent / relative_path.stem
    else:
        # Flatten structure - all files go to destination root
        dest_path = dest_dir / source_file.stem
    
    # Create directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    return dest_path

def detect_metadata(image: Image.Image, file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Detect and catalog all metadata in an image"""
    metadata_info = {
        "has_exif": False,
        "has_icc_profile": False,
        "has_xmp": False,
        "has_iptc": False,
        "exif_data": {},
        "sensitive_data_found": [],
        "total_metadata_size": 0
    }
    
    # Check for EXIF data
    if hasattr(image, '_getexif') and image._getexif() is not None:
        metadata_info["has_exif"] = True
        try:
            exif_dict = piexif.load(image.info.get("exif", b""))
            metadata_info["exif_data"] = exif_dict
        except:
            pass
    
    # Check image info for various metadata
    if image.info:
        for key, value in image.info.items():
            if key == "icc_profile":
                metadata_info["has_icc_profile"] = True
                metadata_info["total_metadata_size"] += len(value) if isinstance(value, bytes) else 0
            elif key in ["xmp", "xml:com.adobe.xmp"]:
                metadata_info["has_xmp"] = True
                metadata_info["total_metadata_size"] += len(str(value))
            elif "iptc" in key.lower():
                metadata_info["has_iptc"] = True
                metadata_info["total_metadata_size"] += len(str(value))
    
    # Detect sensitive information in EXIF
    if metadata_info["has_exif"]:
        sensitive_tags = {
            "GPS": ["GPS coordinates", "GPS location data"],
            "DateTime": ["Creation date/time", "Photo timestamp"],
            "Make": ["Camera manufacturer"],
            "Model": ["Camera model"],
            "Software": ["Software used"],
            "Artist": ["Author/creator information"], 
            "Copyright": ["Copyright information"],
            "UserComment": ["User comments"],
            "ImageDescription": ["Image description"],
            "XPComment": ["Windows XP comment"],
            "XPAuthor": ["Windows XP author"],
            "XPKeywords": ["Windows XP keywords"],
            "XPSubject": ["Windows XP subject"]
        }
        
        for tag_category, descriptions in sensitive_tags.items():
            if any(tag_category.lower() in str(key).lower() for key in metadata_info["exif_data"]):
                metadata_info["sensitive_data_found"].extend(descriptions)
    
    return metadata_info

def strip_comprehensive_metadata(
    image: Image.Image, 
    preserve_color_profile: bool = False,
    preserve_copyright: bool = False,
    add_copyright: Optional[str] = None,
    create_log: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Comprehensively remove metadata while preserving specified elements
    
    Returns:
        Tuple of (clean_image, removal_log)
    """
    
    # Detect existing metadata
    metadata_detected = detect_metadata(image)
    
    removal_log = {
        "original_metadata": metadata_detected,
        "removed_items": [],
        "preserved_items": [],
        "added_items": [],
        "size_reduction": 0
    }
    
    # Create clean image without metadata
    if image.mode == 'P':
        # Handle palette mode specially
        clean_image = image.copy()
        clean_image.info = {}
    else:
        # For other modes, recreate image to strip metadata
        data = list(image.getdata())
        clean_image = Image.new(image.mode, image.size)
        clean_image.putdata(data)
    
    # Preserve ICC color profile if requested
    if preserve_color_profile and 'icc_profile' in image.info:
        clean_image.info['icc_profile'] = image.info['icc_profile']
        removal_log["preserved_items"].append("ICC color profile")
    else:
        if 'icc_profile' in image.info:
            removal_log["removed_items"].append("ICC color profile")
    
    # Handle copyright preservation/addition
    if preserve_copyright or add_copyright:
        try:
            # Create minimal EXIF with only copyright
            exif_dict = piexif.load(image.info.get("exif", b""))
            clean_exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            if preserve_copyright and piexif.ImageIFD.Copyright in exif_dict.get("0th", {}):
                clean_exif["0th"][piexif.ImageIFD.Copyright] = exif_dict["0th"][piexif.ImageIFD.Copyright]
                removal_log["preserved_items"].append("Original copyright information")
            
            if add_copyright:
                clean_exif["0th"][piexif.ImageIFD.Copyright] = add_copyright.encode('utf-8')
                removal_log["added_items"].append(f"Copyright notice: {add_copyright}")
            
            if clean_exif["0th"]:  # Only add EXIF if there's something to preserve/add
                exif_bytes = piexif.dump(clean_exif)
                clean_image.info["exif"] = exif_bytes
        except Exception as e:
            removal_log["removed_items"].append(f"Copyright handling error: {str(e)}")
    
    # Log what was removed
    if metadata_detected["has_exif"]:
        removal_log["removed_items"].extend([
            "EXIF metadata (camera settings, GPS, timestamps)",
            "Camera make and model information",
            "Photo creation date/time",
            "GPS coordinates and location data",
            "Camera settings (ISO, aperture, etc.)",
            "Software information",
            "Thumbnail images",
            "User comments and descriptions"
        ])
    
    if metadata_detected["has_xmp"]:
        removal_log["removed_items"].append("XMP metadata (Adobe software metadata)")
    
    if metadata_detected["has_iptc"]:
        removal_log["removed_items"].append("IPTC metadata (news/publishing metadata)")
    
    # Calculate size reduction
    original_size = metadata_detected["total_metadata_size"]
    remaining_size = len(clean_image.info.get("exif", b"")) + len(clean_image.info.get("icc_profile", b""))
    removal_log["size_reduction"] = max(0, original_size - remaining_size)
    
    return clean_image, removal_log

def sanitize_metadata(
    image: Image.Image,
    remove_gps: bool = True,
    remove_timestamps: bool = True,
    remove_camera_info: bool = True,
    remove_software_info: bool = True,
    preserve_copyright: bool = True,
    preserve_color_profile: bool = True
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Selectively remove sensitive metadata while preserving useful information
    """
    metadata_detected = detect_metadata(image)
    sanitization_log = {
        "original_metadata": metadata_detected,
        "removed_items": [],
        "preserved_items": [],
        "sanitized": True
    }
    
    try:
        # Load existing EXIF
        exif_dict = piexif.load(image.info.get("exif", b""))
        sanitized_exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # Selectively copy non-sensitive data
        for ifd_name, ifd_dict in exif_dict.items():
            if ifd_name == "thumbnail":
                continue
                
            for tag, value in ifd_dict.items():
                remove_tag = False
                reason = ""
                
                # Check GPS data
                if ifd_name == "GPS" and remove_gps:
                    remove_tag = True
                    reason = "GPS location data"
                
                # Check timestamps
                elif remove_timestamps and tag in [
                    piexif.ImageIFD.DateTime,
                    piexif.ExifIFD.DateTimeOriginal,
                    piexif.ExifIFD.DateTimeDigitized
                ]:
                    remove_tag = True
                    reason = "timestamp information"
                
                # Check camera info
                elif remove_camera_info and tag in [
                    piexif.ImageIFD.Make,
                    piexif.ImageIFD.Model,
                    piexif.ExifIFD.LensMake,
                    piexif.ExifIFD.LensModel
                ]:
                    remove_tag = True
                    reason = "camera identification"
                
                # Check software info
                elif remove_software_info and tag in [
                    piexif.ImageIFD.Software,
                    piexif.ImageIFD.ProcessingSoftware
                ]:
                    remove_tag = True
                    reason = "software information"
                
                # Preserve copyright if requested
                elif tag == piexif.ImageIFD.Copyright and preserve_copyright:
                    sanitized_exif[ifd_name][tag] = value
                    sanitization_log["preserved_items"].append("Copyright information")
                    continue
                
                if remove_tag:
                    sanitization_log["removed_items"].append(reason)
                else:
                    # Keep non-sensitive technical data
                    if tag not in [piexif.ImageIFD.XPComment, piexif.ImageIFD.XPAuthor,
                                  piexif.ImageIFD.XPKeywords, piexif.ImageIFD.ImageDescription]:
                        sanitized_exif[ifd_name][tag] = value
        
        # Create clean image
        clean_image = image.copy()
        clean_image.info = {}
        
        # Add sanitized EXIF
        if any(sanitized_exif[ifd] for ifd in ["0th", "Exif", "GPS", "1st"]):
            exif_bytes = piexif.dump(sanitized_exif)
            clean_image.info["exif"] = exif_bytes
        
        # Preserve ICC profile if requested
        if preserve_color_profile and 'icc_profile' in image.info:
            clean_image.info['icc_profile'] = image.info['icc_profile']
            sanitization_log["preserved_items"].append("ICC color profile")
        
        return clean_image, sanitization_log
        
    except Exception as e:
        # Fallback to complete stripping if sanitization fails
        sanitization_log["removed_items"].append(f"Sanitization failed, performed complete removal: {str(e)}")
        clean_image, removal_log = strip_comprehensive_metadata(image, preserve_color_profile)
        return clean_image, sanitization_log

# Legacy function for backward compatibility  
def strip_image_metadata(image: Image.Image) -> Image.Image:
    """Basic metadata removal for backward compatibility"""
    clean_image, _ = strip_comprehensive_metadata(image, preserve_color_profile=False)
    return clean_image

# Protection presets for different levels of adversarial noise
PROTECTION_PRESETS = {
    'light': {
        'intensity': 0.02,
        'method': 'gaussian',
        'description': 'Minimal protection with virtually no visual impact'
    },
    'medium': {
        'intensity': 0.05,
        'method': 'perlin',
        'description': 'Balanced protection with natural-looking noise patterns'
    },
    'strong': {
        'intensity': 0.08,
        'method': 'adversarial',
        'description': 'Maximum protection with advanced perturbations'
    },
    'artistic': {
        'intensity': 0.06,
        'method': 'style_specific',
        'description': 'Tailored protection for artwork and creative content'
    }
}

def generate_perlin_noise(shape: Tuple[int, int, int], scale: float = 10.0, octaves: int = 6, 
                         persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Generate Perlin noise for more natural-looking adversarial patterns"""
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(t, a, b):
        return a + t * (b - a)
    
    def grad(hash_val, x, y):
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else 0)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    # Simplified Perlin noise implementation
    height, width, channels = shape
    noise = np.zeros((height, width))
    
    for octave in range(octaves):
        freq = scale * (lacunarity ** octave)
        amp = persistence ** octave
        
        # Generate octave
        for i in range(height):
            for j in range(width):
                x = j / freq
                y = i / freq
                
                # Get integer coordinates
                xi = int(x) & 255
                yi = int(y) & 255
                
                # Get fractional coordinates
                xf = x - int(x)
                yf = y - int(y)
                
                # Fade curves
                u = fade(xf)
                v = fade(yf)
                
                # Hash coordinates
                aa = (xi + yi * 57) % 256
                ab = (xi + (yi + 1) * 57) % 256
                ba = ((xi + 1) + yi * 57) % 256
                bb = ((xi + 1) + (yi + 1) * 57) % 256
                
                # Calculate gradients
                x1 = lerp(u, grad(aa, xf, yf), grad(ba, xf - 1, yf))
                x2 = lerp(u, grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1))
                
                noise[i, j] += lerp(v, x1, x2) * amp
    
    # Normalize and expand to all channels
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = (noise - 0.5) * 2  # Center around 0
    
    return np.stack([noise] * channels, axis=2)

def generate_frequency_based_noise(image_array: np.ndarray, intensity: float) -> np.ndarray:
    """Generate noise that targets specific frequency bands"""
    noise = np.zeros_like(image_array, dtype=np.float32)
    
    # Process each channel separately
    for c in range(image_array.shape[2]):
        channel = image_array[:, :, c].astype(np.float32)
        
        # Apply FFT
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        
        # Create frequency mask (focus on mid-frequencies)
        h, w = channel.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Focus on mid-frequencies (avoid very low and very high)
        freq_mask = np.exp(-((dist - h * 0.15)**2) / (2 * (h * 0.1)**2))
        freq_mask += np.exp(-((dist - h * 0.3)**2) / (2 * (h * 0.1)**2))
        
        # Generate noise in frequency domain
        noise_freq = np.random.normal(0, intensity, f_shifted.shape) * freq_mask
        
        # Apply noise to frequency domain
        modified_freq = f_shifted + noise_freq
        
        # Convert back to spatial domain
        f_ishifted = np.fft.ifftshift(modified_freq)
        modified_channel = np.fft.ifft2(f_ishifted)
        
        noise[:, :, c] = np.real(modified_channel) - channel
    
    return noise

def generate_adversarial_pattern(image_array: np.ndarray, intensity: float) -> np.ndarray:
    """Generate targeted adversarial perturbations"""
    # Edge detection for targeting high-frequency areas
    gray = np.mean(image_array, axis=2)
    edges = ndimage.sobel(gray)
    
    # Normalize edges (handle edge case where max is 0)
    max_edge = np.max(edges)
    if max_edge > 0:
        edges = edges / max_edge
    else:
        edges = np.zeros_like(edges)
    
    # Generate base noise
    noise = np.random.normal(0, intensity * 255, image_array.shape)
    
    # Apply edge-based weighting (stronger noise near edges)
    edge_weight = np.stack([edges] * image_array.shape[2], axis=2)
    adaptive_noise = noise * (1 + edge_weight * 2)
    
    # Add spiral pattern for additional disruption
    h, w = image_array.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Create spiral pattern
    angle = np.arctan2(y - center_y, x - center_x)
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    spiral = np.sin(angle * 5 + radius * 0.1) * intensity * 50
    
    # Combine patterns
    for c in range(image_array.shape[2]):
        adaptive_noise[:, :, c] += spiral
    
    return adaptive_noise

def generate_style_specific_noise(image_array: np.ndarray, intensity: float) -> np.ndarray:
    """Generate noise specifically designed to protect artistic styles"""
    # Analyze image characteristics
    mean_brightness = np.mean(image_array)
    contrast = np.std(image_array)
    
    # Base noise with artistic considerations
    noise = np.random.normal(0, intensity * 255, image_array.shape)
    
    # Add texture-based modifications
    for c in range(image_array.shape[2]):
        channel = image_array[:, :, c]
        
        # Apply different noise characteristics based on brightness
        bright_mask = channel > mean_brightness
        dark_mask = channel <= mean_brightness
        
        # Stronger protection in smooth areas, subtle in textured areas
        texture_variance = ndimage.generic_filter(channel, np.var, size=3)
        texture_weight = 1 / (1 + texture_variance / 100)
        
        noise[:, :, c] *= texture_weight
        
        # Color shift protection (subtle hue changes)
        if c < 3:  # RGB channels
            color_shift = np.sin(channel / 255 * np.pi * 2) * intensity * 30
            noise[:, :, c] += color_shift
    
    return noise

def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def add_adversarial_noise(image: Image.Image, intensity: float = 0.05, 
                         method: str = 'gaussian', protection_level: str = 'medium',
                         preview_noise: bool = False) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Add sophisticated adversarial noise to protect against AI training
    
    Args:
        image: Input PIL Image
        intensity: Noise intensity (0.01-0.1)
        method: Noise generation method
        protection_level: Protection preset level
        preview_noise: Whether to return noise visualization
    
    Returns:
        Tuple of (protected_image, protection_info)
    """
    
    # Apply protection preset if specified
    if protection_level in PROTECTION_PRESETS:
        preset = PROTECTION_PRESETS[protection_level]
        if method == 'auto':
            method = preset['method']
        intensity = max(intensity, preset['intensity'])  # Use higher of preset or user setting
    
    protection_info = {
        'method': method,
        'intensity': intensity,
        'protection_level': protection_level,
        'psnr': 0,
        'estimated_effectiveness': 'Unknown',
        'visual_impact': 'Minimal'
    }
    
    # Handle transparency
    has_alpha = image.mode in ('RGBA', 'LA')
    if has_alpha:
        # Process RGB channels, preserve alpha
        alpha = image.split()[-1]
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=alpha)
        img_array = np.array(rgb_image, dtype=np.float32)
    else:
        img_array = np.array(image, dtype=np.float32)
    
    original_array = img_array.copy()
    
    # Generate noise based on method
    if method == 'gaussian':
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        protection_info['estimated_effectiveness'] = 'Moderate'
        
    elif method == 'perlin':
        noise = generate_perlin_noise(img_array.shape, scale=20.0) * intensity * 255
        protection_info['estimated_effectiveness'] = 'Good'
        
    elif method == 'frequency':
        noise = generate_frequency_based_noise(img_array, intensity)
        protection_info['estimated_effectiveness'] = 'Very Good'
        
    elif method == 'adversarial':
        noise = generate_adversarial_pattern(img_array, intensity)
        protection_info['estimated_effectiveness'] = 'Excellent'
        
    elif method == 'style_specific':
        noise = generate_style_specific_noise(img_array, intensity)
        protection_info['estimated_effectiveness'] = 'Excellent (Artistic)'
        
    else:  # Default to gaussian
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        protection_info['estimated_effectiveness'] = 'Moderate'
    
    # Apply noise with clipping
    protected_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Calculate quality metrics
    psnr = calculate_psnr(original_array, protected_array.astype(np.float32))
    protection_info['psnr'] = round(psnr, 2)
    
    # Determine visual impact
    if psnr > 45:
        protection_info['visual_impact'] = 'Imperceptible'
    elif psnr > 40:
        protection_info['visual_impact'] = 'Minimal'
    elif psnr > 35:
        protection_info['visual_impact'] = 'Slight'
    else:
        protection_info['visual_impact'] = 'Noticeable'
    
    # Convert back to PIL Image
    protected_image = Image.fromarray(protected_array)
    
    # Handle alpha channel
    if has_alpha:
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        result.paste(protected_image, mask=alpha)
        protected_image = result
    
    # Create noise preview if requested
    if preview_noise:
        noise_normalized = np.clip((noise + 127.5), 0, 255).astype(np.uint8)
        protection_info['noise_preview'] = Image.fromarray(noise_normalized)
    
    return protected_image, protection_info

# Legacy function for backward compatibility
def add_adversarial_noise_legacy(image: Image.Image, intensity: float = 0.01) -> Image.Image:
    """Legacy function for backward compatibility"""
    protected_image, _ = add_adversarial_noise(image, intensity, method='gaussian')
    return protected_image

def add_watermark(
    image: Image.Image,
    watermark_text: Optional[str] = None,
    watermark_image: Optional[Image.Image] = None,
    position: str = 'bottom-right',
    opacity: float = 0.3,
    font_size: Optional[int] = None,
    font_color: str = 'white'
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Add text or image watermark with customizable placement
    
    Args:
        image: Input PIL Image
        watermark_text: Text to use as watermark
        watermark_image: Image to use as watermark
        position: Watermark position (center, top-left, top-right, bottom-left, bottom-right)
        opacity: Watermark opacity (0.0-1.0)
        font_size: Font size for text watermark (auto-calculated if None)
        font_color: Color for text watermark
    
    Returns:
        Tuple of (watermarked_image, watermark_info)
    """
    
    watermark_info = {
        "type": "none",
        "position": position,
        "opacity": opacity,
        "applied": False
    }
    
    if not watermark_text and not watermark_image:
        return image, watermark_info
    
    # Create a copy to work with
    watermarked = image.copy()
    
    # Convert to RGBA if needed for transparency
    if watermarked.mode != 'RGBA':
        watermarked = watermarked.convert('RGBA')
    
    # Create transparent overlay
    overlay = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
    
    if watermark_text:
        # Text watermark
        watermark_info["type"] = "text"
        watermark_info["text"] = watermark_text
        
        # Calculate font size if not provided
        if font_size is None:
            # Base font size on image dimensions
            font_size = max(20, min(watermarked.width, watermarked.height) // 25)
        
        try:
            # Try to use a better font if available
            from PIL import ImageFont, ImageDraw
            
            try:
                # Try common system fonts
                font_paths = [
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/Windows/Fonts/arial.ttf",         # Windows
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
            
            draw = ImageDraw.Draw(overlay)
            
            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            margin = min(watermarked.width, watermarked.height) // 40
            
            if position == 'center':
                x = (watermarked.width - text_width) // 2
                y = (watermarked.height - text_height) // 2
            elif position == 'top-left':
                x, y = margin, margin
            elif position == 'top-right':
                x = watermarked.width - text_width - margin
                y = margin
            elif position == 'bottom-left':
                x = margin
                y = watermarked.height - text_height - margin
            else:  # bottom-right (default)
                x = watermarked.width - text_width - margin
                y = watermarked.height - text_height - margin
            
            # Convert color name to RGB
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'red': (255, 0, 0),
                'blue': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255)
            }
            
            color_rgb = color_map.get(font_color.lower(), (255, 255, 255))
            alpha = int(opacity * 255)
            text_color = color_rgb + (alpha,)
            
            # Draw text with slight shadow for better visibility
            shadow_offset = max(1, font_size // 20)
            draw.text((x + shadow_offset, y + shadow_offset), watermark_text, 
                     font=font, fill=(0, 0, 0, alpha // 2))  # Shadow
            draw.text((x, y), watermark_text, font=font, fill=text_color)  # Main text
            
            watermark_info["font_size"] = font_size
            watermark_info["color"] = font_color
            
        except ImportError:
            # Fallback if PIL doesn't have font support
            watermark_info["error"] = "Font support not available"
            return image, watermark_info
    
    elif watermark_image:
        # Image watermark
        watermark_info["type"] = "image"
        
        # Resize watermark image to appropriate size
        max_watermark_size = min(watermarked.width // 4, watermarked.height // 4)
        watermark_resized = watermark_image.copy()
        
        # Maintain aspect ratio
        if watermark_resized.width > max_watermark_size or watermark_resized.height > max_watermark_size:
            watermark_resized.thumbnail((max_watermark_size, max_watermark_size), Image.Resampling.LANCZOS)
        
        # Convert to RGBA
        if watermark_resized.mode != 'RGBA':
            watermark_resized = watermark_resized.convert('RGBA')
        
        # Apply opacity
        watermark_with_opacity = Image.new('RGBA', watermark_resized.size, (0, 0, 0, 0))
        for y in range(watermark_resized.height):
            for x in range(watermark_resized.width):
                r, g, b, a = watermark_resized.getpixel((x, y))
                new_alpha = int(a * opacity)
                watermark_with_opacity.putpixel((x, y), (r, g, b, new_alpha))
        
        # Calculate position
        margin = min(watermarked.width, watermarked.height) // 40
        wm_width, wm_height = watermark_with_opacity.size
        
        if position == 'center':
            x = (watermarked.width - wm_width) // 2
            y = (watermarked.height - wm_height) // 2
        elif position == 'top-left':
            x, y = margin, margin
        elif position == 'top-right':
            x = watermarked.width - wm_width - margin
            y = margin
        elif position == 'bottom-left':
            x = margin
            y = watermarked.height - wm_height - margin
        else:  # bottom-right (default)
            x = watermarked.width - wm_width - margin
            y = watermarked.height - wm_height - margin
        
        # Paste watermark
        overlay.paste(watermark_with_opacity, (x, y), watermark_with_opacity)
        
        watermark_info["watermark_size"] = watermark_with_opacity.size
    
    # Composite the overlay with the original image
    final_image = Image.alpha_composite(watermarked, overlay)
    
    # Convert back to original mode if it wasn't RGBA
    if image.mode != 'RGBA':
        final_image = final_image.convert(image.mode)
    
    watermark_info["applied"] = True
    return final_image, watermark_info

def generate_filename(
    original_name: str,
    pattern: str = "{original}",
    counter: int = 1,
    image_size: Optional[Tuple[int, int]] = None,
    file_extension: str = "",
    timestamp: Optional[datetime] = None
) -> str:
    """
    Generate new filename based on pattern
    
    Available patterns:
    - {original}: Original filename without extension
    - {number}: Sequential number with padding
    - {date}: Current date (YYYY-MM-DD)
    - {time}: Current time (HH-MM-SS)
    - {datetime}: Full datetime (YYYY-MM-DD_HH-MM-SS)
    - {width}: Image width
    - {height}: Image height
    - {dimensions}: Image dimensions (WIDTHxHEIGHT)
    """
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Extract base name without extension
    base_name = Path(original_name).stem
    
    # Prepare replacement values
    replacements = {
        'original': base_name,
        'number': f"{counter:04d}",  # 4-digit padding
        'date': timestamp.strftime('%Y-%m-%d'),
        'time': timestamp.strftime('%H-%M-%S'),
        'datetime': timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    # Add image dimensions if available
    if image_size:
        replacements.update({
            'width': str(image_size[0]),
            'height': str(image_size[1]),
            'dimensions': f"{image_size[0]}x{image_size[1]}"
        })
    
    # Apply replacements
    new_name = pattern
    for key, value in replacements.items():
        new_name = new_name.replace(f"{{{key}}}", value)
    
    # Clean filename for web safety
    new_name = re.sub(r'[^\w\-_.]', '_', new_name)
    new_name = re.sub(r'_+', '_', new_name)  # Replace multiple underscores
    
    # Add extension
    if file_extension and not new_name.endswith(file_extension):
        new_name += file_extension
    
    return new_name

def generate_processing_report(
    source_dir: str,
    dest_dir: str,
    results: List[Dict[str, Any]],
    start_time: float,
    end_time: float
) -> Dict[str, Any]:
    """Generate comprehensive processing report"""
    
    total_files = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_files - successful
    
    original_sizes = [r.get('original_size_mb', 0) for r in results if r['success']]
    final_sizes = [r.get('final_size_mb', 0) for r in results if r['success']]
    
    total_original_mb = sum(original_sizes)
    total_final_mb = sum(final_sizes)
    space_saved_mb = total_original_mb - total_final_mb
    space_saved_percent = (space_saved_mb / total_original_mb * 100) if total_original_mb > 0 else 0
    
    processing_time = end_time - start_time
    
    # Categorize by processing type
    metadata_stripped = sum(1 for r in results if r.get('metadata_removed', False))
    noise_applied = sum(1 for r in results if r.get('noise_applied', False))
    resized = sum(1 for r in results if r.get('resized', False))
    watermarked = sum(1 for r in results if r.get('watermarked', False))
    
    # Quality metrics
    avg_psnr = np.mean([r.get('psnr', 0) for r in results if r.get('psnr', 0) > 0])
    
    report = {
        'summary': {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_files * 100) if total_files > 0 else 0,
            'processing_time_seconds': round(processing_time, 2),
            'avg_time_per_file': round(processing_time / max(total_files, 1), 2)
        },
        'file_sizes': {
            'original_total_mb': round(total_original_mb, 2),
            'final_total_mb': round(total_final_mb, 2),
            'space_saved_mb': round(space_saved_mb, 2),
            'space_saved_percent': round(space_saved_percent, 2),
            'avg_original_size_mb': round(np.mean(original_sizes), 2) if original_sizes else 0,
            'avg_final_size_mb': round(np.mean(final_sizes), 2) if final_sizes else 0
        },
        'processing_stats': {
            'metadata_stripped': metadata_stripped,
            'noise_protection_applied': noise_applied,
            'images_resized': resized,
            'watermarks_added': watermarked,
            'avg_quality_psnr': round(avg_psnr, 2) if not np.isnan(avg_psnr) else 0
        },
        'directories': {
            'source': source_dir,
            'destination': dest_dir
        },
        'timestamp': datetime.now().isoformat(),
        'detailed_results': results
    }
    
    return report

def export_report_csv(report: Dict[str, Any], output_path: str) -> bool:
    """Export processing report to CSV format"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Summary section
            writer.writerow(['Processing Report Summary'])
            writer.writerow(['Generated:', report['timestamp']])
            writer.writerow([''])
            
            writer.writerow(['Metric', 'Value'])
            for key, value in report['summary'].items():
                writer.writerow([key.replace('_', ' ').title(), value])
            
            writer.writerow([''])
            writer.writerow(['File Size Analysis'])
            for key, value in report['file_sizes'].items():
                writer.writerow([key.replace('_', ' ').title(), value])
            
            writer.writerow([''])
            writer.writerow(['Processing Statistics'])
            for key, value in report['processing_stats'].items():
                writer.writerow([key.replace('_', ' ').title(), value])
            
            # Detailed results
            if report.get('detailed_results'):
                writer.writerow([''])
                writer.writerow(['Detailed File Results'])
                
                # Header
                sample_result = report['detailed_results'][0]
                headers = ['Filename', 'Success', 'Original Size (MB)', 'Final Size (MB)', 'Message']
                writer.writerow(headers)
                
                # Data rows
                for result in report['detailed_results']:
                    row = [
                        result.get('filename', ''),
                        result.get('success', False),
                        result.get('original_size_mb', 0),
                        result.get('final_size_mb', 0),
                        result.get('message', '')
                    ]
                    writer.writerow(row)
        
        return True
    except Exception as e:
        logging.error(f"Failed to export CSV report: {e}")
        return False

def auto_enhance_image(image: Image.Image, enhance_level: str = 'subtle') -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Apply automatic enhancements to improve image quality
    
    Args:
        image: Input PIL Image
        enhance_level: Enhancement level ('subtle', 'moderate', 'strong')
    
    Returns:
        Tuple of (enhanced_image, enhancement_info)
    """
    
    enhancement_levels = {
        'subtle': {'contrast': 1.1, 'brightness': 1.05, 'sharpness': 1.1, 'color': 1.05},
        'moderate': {'contrast': 1.2, 'brightness': 1.1, 'sharpness': 1.2, 'color': 1.1},
        'strong': {'contrast': 1.3, 'brightness': 1.15, 'sharpness': 1.3, 'color': 1.15}
    }
    
    factors = enhancement_levels.get(enhance_level, enhancement_levels['subtle'])
    
    enhanced = image.copy()
    applied_enhancements = []
    
    try:
        # Apply contrast enhancement
        if factors['contrast'] != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(factors['contrast'])
            applied_enhancements.append(f"Contrast: {factors['contrast']:.2f}")
        
        # Apply brightness enhancement
        if factors['brightness'] != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(factors['brightness'])
            applied_enhancements.append(f"Brightness: {factors['brightness']:.2f}")
        
        # Apply sharpness enhancement
        if factors['sharpness'] != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(factors['sharpness'])
            applied_enhancements.append(f"Sharpness: {factors['sharpness']:.2f}")
        
        # Apply color enhancement (saturation)
        if factors['color'] != 1.0 and enhanced.mode in ('RGB', 'RGBA'):
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(factors['color'])
            applied_enhancements.append(f"Saturation: {factors['color']:.2f}")
        
        enhancement_info = {
            'level': enhance_level,
            'applied': applied_enhancements,
            'success': True
        }
        
    except Exception as e:
        enhancement_info = {
            'level': enhance_level,
            'applied': [],
            'success': False,
            'error': str(e)
        }
        enhanced = image  # Return original on error
    
    return enhanced, enhancement_info

def calculate_file_size_mb(image: Image.Image) -> float:
    """Estimate image file size in MB"""
    width, height = image.size
    channels = len(image.getbands())
    # Rough estimate: uncompressed size / compression ratio
    bytes_per_pixel = channels
    total_pixels = width * height
    uncompressed_mb = (total_pixels * bytes_per_pixel) / (1024 * 1024)
    # Assume average compression ratio of 10:1 for JPEG
    return uncompressed_mb / 10

def apply_sharpening(image: Image.Image, amount: float = 0.2) -> Image.Image:
    """Apply subtle sharpening to maintain clarity after resizing"""
    if amount <= 0:
        return image
    
    # Create unsharp mask
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    sharpened = ImageEnhance.Sharpness(image).enhance(1.0 + amount)
    return sharpened

def smart_resize_for_web(
    image: Image.Image, 
    max_width: int, 
    max_height: int,
    retina_mode: bool = False,
    aggressive_optimization: bool = False,
    maintain_quality: bool = True
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Advanced image resizing optimized for web with quality preservation
    
    Returns:
        Tuple of (resized_image, resize_info)
    """
    original_width, original_height = image.size
    original_size_mb = calculate_file_size_mb(image)
    
    resize_info = {
        "original_size": (original_width, original_height),
        "original_size_mb": original_size_mb,
        "resized": False,
        "scale_factor": 1.0,
        "sharpening_applied": False,
        "optimization_applied": False
    }
    
    # Apply retina mode (2x dimensions)
    target_width = max_width * 2 if retina_mode else max_width
    target_height = max_height * 2 if retina_mode else max_height
    
    # For very large files, apply more aggressive optimization
    if aggressive_optimization or original_size_mb > 5.0:
        target_width = int(target_width * 0.8)
        target_height = int(target_height * 0.8)
        resize_info["optimization_applied"] = True
    
    # Calculate scaling factor maintaining aspect ratio
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale unless retina
    
    # Allow upscaling only for retina mode
    if retina_mode and scale_factor > 1.0:
        scale_factor = min(width_ratio, height_ratio, 2.0)  # Max 2x upscale
    
    if scale_factor < 0.99:  # Only resize if significant change
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Use high-quality resampling
        resampling_method = Image.Resampling.LANCZOS
        if scale_factor < 0.5:  # For significant downscaling
            # Use bicubic for very small scaling
            resampling_method = Image.Resampling.BICUBIC
        
        resized_image = image.resize((new_width, new_height), resampling_method)
        resize_info["resized"] = True
        resize_info["scale_factor"] = scale_factor
        resize_info["new_size"] = (new_width, new_height)
        
        # Apply subtle sharpening for downscaled images if quality maintenance is enabled
        if maintain_quality and scale_factor < 0.8:
            sharpening_amount = min(0.3, (1.0 - scale_factor) * 0.5)
            resized_image = apply_sharpening(resized_image, sharpening_amount)
            resize_info["sharpening_applied"] = True
            resize_info["sharpening_amount"] = sharpening_amount
        
        return resized_image, resize_info
    
    return image, resize_info

def get_preset_dimensions(preset_name: str) -> Tuple[int, int]:
    """Get dimensions for a preset"""
    return DIMENSION_PRESETS.get(preset_name, (1920, 1080))

def optimize_for_web(image: Image.Image, original_format: str, file_size_mb: float) -> Dict[str, Any]:
    """Choose best format and settings for web optimization"""
    has_transparency = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    is_photo = original_format.upper() in ['JPEG', 'JPG'] and not has_transparency
    is_large_file = file_size_mb > 2.0
    
    optimization_info = {
        "recommended_format": None,
        "quality_setting": 85,
        "reasoning": "",
        "save_kwargs": {}
    }
    
    if has_transparency:
        # PNG for transparency, WebP if large
        if is_large_file:
            optimization_info["recommended_format"] = "WEBP"
            optimization_info["quality_setting"] = 90
            optimization_info["reasoning"] = "WebP for large file with transparency"
            optimization_info["save_kwargs"] = {
                "quality": 90,
                "lossless": False,
                "method": 6
            }
        else:
            optimization_info["recommended_format"] = "PNG"
            optimization_info["reasoning"] = "PNG to preserve transparency"
            optimization_info["save_kwargs"] = {
                "optimize": True,
                "compress_level": 6
            }
    elif is_photo:
        # Photograph - use JPEG or WebP for large files
        if is_large_file:
            optimization_info["recommended_format"] = "WEBP"
            optimization_info["quality_setting"] = 85
            optimization_info["reasoning"] = "WebP for large photograph"
            optimization_info["save_kwargs"] = {
                "quality": 85,
                "lossless": False,
                "method": 6
            }
        else:
            optimization_info["recommended_format"] = "JPEG"
            optimization_info["quality_setting"] = 90
            optimization_info["reasoning"] = "JPEG for photograph"
            optimization_info["save_kwargs"] = {
                "quality": 90,
                "progressive": True,
                "optimize": True
            }
    else:
        # Graphics/illustrations
        if is_large_file:
            optimization_info["recommended_format"] = "WEBP"
            optimization_info["quality_setting"] = 95
            optimization_info["reasoning"] = "WebP for large graphics"
            optimization_info["save_kwargs"] = {
                "quality": 95,
                "lossless": True
            }
        else:
            optimization_info["recommended_format"] = "PNG"
            optimization_info["reasoning"] = "PNG for graphics/illustrations"
            optimization_info["save_kwargs"] = {
                "optimize": True,
                "compress_level": 6
            }
    
    return optimization_info

def determine_output_format(input_format: str, output_format: str, has_transparency: bool, file_size_mb: float = 0) -> str:
    """Determine the best output format based on input and settings"""
    if output_format.upper() != "AUTO":
        return output_format.upper()
    
    # Use smart optimization for auto mode
    dummy_image = Image.new('RGBA' if has_transparency else 'RGB', (100, 100))
    optimization = optimize_for_web(dummy_image, input_format, file_size_mb)
    
    return optimization["recommended_format"]

def process_single_image(
    source_file: Path, 
    source_dir: Path, 
    dest_dir: Path,
    resize_enabled: bool,
    strip_metadata: bool,
    add_noise: bool,
    preserve_structure: bool,
    output_format: str,
    quality: int,
    max_width: int,
    max_height: int,
    retina_mode: bool = False,
    aggressive_optimization: bool = False,
    maintain_quality: bool = True,
    preserve_color_profile: bool = True,
    preserve_copyright: bool = False,
    add_copyright: Optional[str] = None,
    create_metadata_log: bool = False,
    metadata_mode: str = "complete",
    noise_intensity: float = 0.05,
    noise_method: str = "gaussian",
    protection_level: str = "medium",
    preview_noise: bool = False,
    add_watermark_enabled: bool = False,
    watermark_text: str = "",
    watermark_position: str = "bottom-right",
    watermark_opacity: float = 0.3,
    auto_enhance: bool = False,
    enhance_level: str = "subtle",
    filename_pattern: str = "{original}",
    counter: int = 1
) -> Tuple[bool, str, Dict[str, Any]]:
    """Process a single image file"""
    try:
        # Load image
        with Image.open(source_file) as image:
            # Calculate original file size
            original_size_mb = calculate_file_size_mb(image)
            
            # Convert to RGB if necessary for processing
            if image.mode in ('RGBA', 'LA'):
                has_transparency = True
                if strip_metadata or add_noise or resize_enabled:
                    # Keep RGBA for processing
                    processed_image = image.copy()
                else:
                    processed_image = image
            else:
                has_transparency = False
                if image.mode != 'RGB':
                    processed_image = image.convert('RGB')
                else:
                    processed_image = image.copy()
            
            # Store processing information
            processing_info = {
                "original_size": image.size,
                "original_format": source_file.suffix[1:],
                "original_size_mb": original_size_mb
            }
            
            # Apply processing steps
            resize_info = None
            if resize_enabled:
                processed_image, resize_info = smart_resize_for_web(
                    processed_image, 
                    max_width, 
                    max_height,
                    retina_mode=retina_mode,
                    aggressive_optimization=aggressive_optimization,
                    maintain_quality=maintain_quality
                )
                processing_info.update(resize_info)
            
            # Apply metadata stripping with comprehensive options
            metadata_log = None
            if strip_metadata:
                if metadata_mode == "sanitize":
                    processed_image, metadata_log = sanitize_metadata(
                        processed_image,
                        preserve_copyright=preserve_copyright,
                        preserve_color_profile=preserve_color_profile
                    )
                else:  # complete removal
                    processed_image, metadata_log = strip_comprehensive_metadata(
                        processed_image, 
                        preserve_color_profile=preserve_color_profile,
                        preserve_copyright=preserve_copyright,
                        add_copyright=add_copyright,
                        create_log=create_metadata_log
                    )
                processing_info["metadata_removal"] = metadata_log
            
            # Apply adversarial noise protection
            noise_info = None
            if add_noise:
                processed_image, noise_info = add_adversarial_noise(
                    processed_image,
                    intensity=noise_intensity,
                    method=noise_method,
                    protection_level=protection_level,
                    preview_noise=preview_noise
                )
                processing_info["noise_protection"] = noise_info
            
            # Apply auto-enhancement
            enhancement_info = None
            if auto_enhance:
                processed_image, enhancement_info = auto_enhance_image(
                    processed_image,
                    enhance_level=enhance_level
                )
                processing_info["enhancement"] = enhancement_info
            
            # Apply watermark
            watermark_info = None
            if add_watermark_enabled and watermark_text.strip():
                processed_image, watermark_info = add_watermark(
                    processed_image,
                    watermark_text=watermark_text.strip(),
                    position=watermark_position,
                    opacity=watermark_opacity
                )
                processing_info["watermark"] = watermark_info
            
            # Get smart optimization recommendations
            optimization = optimize_for_web(processed_image, source_file.suffix[1:], original_size_mb)
            
            # Determine output format and path
            final_format = determine_output_format(
                source_file.suffix[1:], 
                output_format, 
                has_transparency, 
                original_size_mb
            )
            
            # Generate filename using pattern
            new_filename = generate_filename(
                original_name=source_file.name,
                pattern=filename_pattern,
                counter=counter,
                image_size=processed_image.size,
                file_extension=f".{final_format.lower()}"
            )
            
            dest_path = create_destination_structure(source_file, source_dir, dest_dir, preserve_structure)
            final_path = dest_path.parent / new_filename
            
            # Use the generated filename path
            if final_format == "JPEG":
                # Convert RGBA to RGB for JPEG
                if processed_image.mode == 'RGBA':
                    rgb_image = Image.new('RGB', processed_image.size, (255, 255, 255))
                    rgb_image.paste(processed_image, mask=processed_image.split()[-1])
                    processed_image = rgb_image
            
            # Use smart optimization settings or user settings
            if output_format.upper() == "AUTO":
                save_kwargs = optimization["save_kwargs"].copy()
            else:
                # Use user-specified settings
                save_kwargs = {}
                if final_format in ["JPEG", "WEBP"]:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif final_format == "PNG":
                    save_kwargs['optimize'] = True
            
            processed_image.save(final_path, format=final_format, **save_kwargs)
            
            # Calculate final file size
            final_size_mb = calculate_file_size_mb(processed_image)
            
            # Create comprehensive result info for analytics
            result_info = {
                'filename': new_filename,
                'success': True,
                'message': f"✅ {source_file.name} → {new_filename}",
                'original_size_mb': round(original_size_mb, 3),
                'final_size_mb': round(final_size_mb, 3),
                'metadata_removed': strip_metadata and metadata_log is not None,
                'noise_applied': add_noise and noise_info is not None,
                'resized': resize_enabled and resize_info and resize_info.get("resized", False),
                'watermarked': add_watermark_enabled and watermark_info and watermark_info.get("applied", False),
                'enhanced': auto_enhance and enhancement_info and enhancement_info.get("success", False),
                'psnr': noise_info.get('psnr', 0) if noise_info else 0,
                'processing_info': processing_info
            }
            
            # Create detailed success message
            success_msg = f"✅ {source_file.name}"
            if new_filename != source_file.name:
                success_msg += f" → {new_filename}"
            
            if resize_info and resize_info.get("resized"):
                original_size = resize_info["original_size"]
                new_size = resize_info.get("new_size", original_size)
                success_msg += f" ({original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]})"
            
            if output_format.upper() == "AUTO":
                success_msg += f" | {final_format} ({optimization['reasoning']})"
            
            if metadata_log and strip_metadata:
                removed_count = len(metadata_log.get("removed_items", []))
                preserved_count = len(metadata_log.get("preserved_items", []))
                if removed_count > 0:
                    success_msg += f" | Metadata: {removed_count} removed"
                    if preserved_count > 0:
                        success_msg += f", {preserved_count} preserved"
            
            if enhancement_info and auto_enhance and enhancement_info.get("success"):
                applied_count = len(enhancement_info.get("applied", []))
                success_msg += f" | Enhanced: {applied_count} adjustments"
            
            if watermark_info and add_watermark_enabled and watermark_info.get("applied"):
                success_msg += f" | Watermark: {watermark_info['type']} added"
            
            if noise_info and add_noise:
                success_msg += f" | Protection: {noise_info['estimated_effectiveness']}"
                success_msg += f" (PSNR: {noise_info['psnr']}dB)"
            
            size_reduction = ((original_size_mb - final_size_mb) / original_size_mb * 100) if original_size_mb > 0 else 0
            if size_reduction > 5:  # Only show if significant reduction
                success_msg += f" | Size: -{size_reduction:.1f}%"
            
        return True, success_msg, result_info
    
    except Exception as e:
        error_result = {
            'filename': source_file.name,
            'success': False,
            'message': f"Error processing {source_file.name}: {str(e)}",
            'original_size_mb': 0,
            'final_size_mb': 0,
            'metadata_removed': False,
            'noise_applied': False,
            'resized': False,
            'watermarked': False,
            'enhanced': False,
            'psnr': 0,
            'processing_info': {'error': str(e)}
        }
        return False, f"Error processing {source_file.name}: {str(e)}", error_result

def process_images(
    source_dir: str,
    dest_dir: str,
    resize_enabled: bool,
    strip_metadata: bool,
    add_noise: bool,
    preserve_structure: bool,
    output_format: str,
    quality: int,
    max_width: int,
    max_height: int,
    retina_mode: bool = False,
    aggressive_optimization: bool = False,
    maintain_quality: bool = True,
    dimension_preset: str = "Custom",
    preserve_color_profile: bool = True,
    preserve_copyright: bool = False,
    add_copyright: str = "",
    create_metadata_log: bool = False,
    metadata_mode: str = "complete",
    noise_intensity: float = 0.05,
    noise_method: str = "gaussian",
    protection_level: str = "medium",
    preview_noise: bool = False,
    add_watermark_enabled: bool = False,
    watermark_text: str = "",
    watermark_position: str = "bottom-right",
    watermark_opacity: float = 0.3,
    auto_enhance: bool = False,
    enhance_level: str = "subtle",
    filename_pattern: str = "{original}",
    platform_preset: str = "general",
    generate_report: bool = False,
    progress=gr.Progress()
) -> str:
    """Main processing function"""
    
    # Validation
    if not source_dir or not os.path.exists(source_dir):
        return "❌ Error: Source directory does not exist or is not specified."
    
    if not dest_dir:
        return "❌ Error: Destination directory is not specified."
    
    # Create destination directory
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as e:
        return f"❌ Error: Could not create destination directory: {str(e)}"
    
    # Find all images
    progress(0, desc="Finding images...")
    image_files = find_images(source_dir)
    
    if not image_files:
        return "❌ No supported image files found in the source directory."
    
    # Apply platform preset if specified
    if platform_preset != "general" and platform_preset in PLATFORM_PRESETS:
        preset_config = PLATFORM_PRESETS[platform_preset]
        if dimension_preset == "Custom":  # Only override if user hasn't set custom dimensions
            max_width, max_height = preset_config["max_dimensions"]
        if output_format.upper() == "AUTO":  # Only override if user hasn't set specific format
            output_format = preset_config["format"]
            if "quality" in preset_config:
                quality = preset_config["quality"]
    
    # Use preset dimensions if not custom
    if dimension_preset != "Custom":
        max_width, max_height = get_preset_dimensions(dimension_preset)
    
    # Process images
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    processed_count = 0
    error_count = 0
    error_messages = []
    detailed_results = []
    
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        progress((i + 1) / len(image_files), desc=f"Processing {image_file.name}...")
        
        # Clean add_copyright parameter
        copyright_notice = add_copyright.strip() if add_copyright and add_copyright.strip() else None
        
        success, message, result_info = process_single_image(
            image_file, source_path, dest_path,
            resize_enabled, strip_metadata, add_noise, preserve_structure,
            output_format, quality, max_width, max_height,
            retina_mode, aggressive_optimization, maintain_quality,
            preserve_color_profile, preserve_copyright, copyright_notice,
            create_metadata_log, metadata_mode, noise_intensity, noise_method,
            protection_level, preview_noise, add_watermark_enabled, watermark_text,
            watermark_position, watermark_opacity, auto_enhance, enhance_level,
            filename_pattern, i + 1
        )
        
        detailed_results.append(result_info)
        
        if success:
            processed_count += 1
        else:
            error_count += 1
            error_messages.append(message)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Generate processing report if requested
    report_info = ""
    if generate_report and detailed_results:
        try:
            report = generate_processing_report(source_dir, dest_dir, detailed_results, start_time, end_time)
            
            # Save report as CSV
            report_path = os.path.join(dest_dir, f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if export_report_csv(report, report_path):
                report_info = f"📄 **Processing Report:** `{report_path}`\n"
        except Exception as e:
            report_info = f"⚠️ Report generation failed: {str(e)}\n"
    
    # Calculate detailed analytics
    successful_results = [r for r in detailed_results if r['success']]
    total_original_mb = sum(r.get('original_size_mb', 0) for r in successful_results)
    total_final_mb = sum(r.get('final_size_mb', 0) for r in successful_results)
    space_saved_mb = total_original_mb - total_final_mb
    space_saved_percent = (space_saved_mb / total_original_mb * 100) if total_original_mb > 0 else 0
    
    # Count feature usage
    metadata_stripped_count = sum(1 for r in successful_results if r.get('metadata_removed', False))
    noise_applied_count = sum(1 for r in successful_results if r.get('noise_applied', False))
    resized_count = sum(1 for r in successful_results if r.get('resized', False))
    watermarked_count = sum(1 for r in successful_results if r.get('watermarked', False))
    enhanced_count = sum(1 for r in successful_results if r.get('enhanced', False))
    
    # Build comprehensive result message
    result_lines = [
        "🎉 **Processing Complete!**",
        "",
        f"📊 **Statistics:**",
        f"• Total files found: {len(image_files)}",
        f"• Successfully processed: {processed_count}",
        f"• Errors: {error_count}",
        f"• Processing time: {processing_time:.2f} seconds ({processing_time/max(len(image_files), 1):.2f}s per file)",
        f"• Output directory: `{dest_dir}`",
        ""
    ]
    
    if successful_results:
        result_lines.extend([
            f"💾 **File Size Analysis:**",
            f"• Original total size: {total_original_mb:.2f} MB",
            f"• Final total size: {total_final_mb:.2f} MB",
            f"• Space saved: {space_saved_mb:.2f} MB ({space_saved_percent:.1f}%)",
            ""
        ])
        
        if any([metadata_stripped_count, noise_applied_count, resized_count, watermarked_count, enhanced_count]):
            result_lines.extend([
                f"🔧 **Features Applied:**",
            ])
            if resized_count > 0:
                result_lines.append(f"• Smart resize: {resized_count} images")
            if metadata_stripped_count > 0:
                result_lines.append(f"• Metadata removal: {metadata_stripped_count} images")
            if noise_applied_count > 0:
                result_lines.append(f"• AI protection: {noise_applied_count} images")
            if watermarked_count > 0:
                result_lines.append(f"• Watermarks: {watermarked_count} images")
            if enhanced_count > 0:
                result_lines.append(f"• Auto-enhancement: {enhanced_count} images")
            result_lines.append("")
    
    if report_info:
        result_lines.append(report_info)
    
    if error_count > 0:
        result_lines.append("❌ **Errors encountered:**")
        for error_msg in error_messages[:10]:  # Show first 10 errors
            result_lines.append(f"• {error_msg}")
        if len(error_messages) > 10:
            result_lines.append(f"• ... and {len(error_messages) - 10} more errors")
    
    return "\n".join(result_lines)

# Create Gradio interface
with gr.Blocks(
    title="Image Batch Processor",
    theme=gr.themes.Soft(),
    css="""
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; margin-bottom: 2rem; }
    .section { margin: 1.5rem 0; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px; }
    """
) as app:
    
    gr.HTML("""
    <div class="header">
        <h1>🖼️ Image Batch Processor</h1>
        <p>Professional batch processing tool for images with smart optimization features</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section"><h3>📁 Directory Settings</h3></div>')
            
            source_dir = gr.Textbox(
                label="Source Directory",
                placeholder="Enter path to folder containing images...",
                info="Directory containing images to process"
            )
            
            dest_dir = gr.Textbox(
                label="Destination Directory", 
                placeholder="Enter path for processed images...",
                info="Directory where processed images will be saved"
            )
            
            gr.HTML('<div class="section"><h3>🔧 Processing Options</h3></div>')
            
            with gr.Row():
                resize_enabled = gr.Checkbox(
                    label="Smart Resize for Web",
                    value=True,
                    info="Resize images to specified dimensions while maintaining aspect ratio"
                )
                
                strip_metadata = gr.Checkbox(
                    label="Strip Metadata",
                    value=True,
                    info="Remove EXIF data and other metadata for privacy and smaller file size"
                )
            
            with gr.Row():
                add_noise = gr.Checkbox(
                    label="Add Adversarial Noise",
                    value=False,
                    info="Add subtle noise to protect against AI training (minimal visual impact)"
                )
                
                preserve_structure = gr.Checkbox(
                    label="Preserve Folder Structure",
                    value=True,
                    info="Maintain the original directory structure in output"
                )
            
            gr.HTML('<div class="section"><h3>🔒 Advanced Metadata Options</h3></div>')
            
            metadata_mode = gr.Dropdown(
                choices=["complete", "sanitize"],
                value="complete",
                label="Metadata Removal Mode",
                info="Complete: Remove all metadata | Sanitize: Remove only sensitive data"
            )
            
            with gr.Row():
                preserve_color_profile = gr.Checkbox(
                    label="Preserve Color Profiles",
                    value=True,
                    info="Keep ICC color profiles for accurate color reproduction"
                )
                
                preserve_copyright = gr.Checkbox(
                    label="Preserve Copyright",
                    value=False,
                    info="Keep existing copyright information in metadata"
                )
            
            with gr.Row():
                create_metadata_log = gr.Checkbox(
                    label="Create Metadata Log",
                    value=False,
                    info="Generate detailed log of metadata removal process"
                )
            
            add_copyright = gr.Textbox(
                label="Add Copyright Notice (Optional)",
                placeholder="© Your Name 2024",
                info="Add custom copyright notice to processed images"
            )
            
            gr.HTML('<div class="section"><h3>🛡️ AI Training Protection</h3></div>')
            
            with gr.Row():
                protection_level = gr.Dropdown(
                    choices=["light", "medium", "strong", "artistic"],
                    value="medium",
                    label="Protection Level",
                    info="Preset protection levels with optimized settings"
                )
                
                noise_method = gr.Dropdown(
                    choices=["gaussian", "perlin", "frequency", "adversarial", "style_specific"],
                    value="gaussian",
                    label="Noise Method",
                    info="Different noise generation algorithms for protection"
                )
            
            noise_intensity = gr.Slider(
                minimum=0.01,
                maximum=0.15,
                value=0.05,
                step=0.01,
                label="Noise Intensity",
                info="Higher values = stronger protection but potentially more visible"
            )
            
            with gr.Row():
                preview_noise = gr.Checkbox(
                    label="Preview Noise Pattern",
                    value=False,
                    info="Generate visualization of the noise pattern (for testing)"
                )
            
            # Protection method descriptions
            with gr.Accordion("🔍 Protection Method Details", open=False):
                gr.HTML("""
                <div style="margin: 10px 0;">
                <h4>Protection Methods:</h4>
                <ul>
                    <li><strong>Gaussian:</strong> Basic random noise - good for general protection</li>
                    <li><strong>Perlin:</strong> Natural-looking noise patterns - less detectable</li>
                    <li><strong>Frequency:</strong> Targets specific frequency bands - advanced protection</li>
                    <li><strong>Adversarial:</strong> Edge-focused perturbations - maximum effectiveness</li>
                    <li><strong>Style-specific:</strong> Tailored for artwork protection - preserves artistic intent</li>
                </ul>
                <h4>Protection Levels:</h4>
                <ul>
                    <li><strong>Light:</strong> Minimal protection, virtually imperceptible</li>
                    <li><strong>Medium:</strong> Balanced protection with natural patterns</li>
                    <li><strong>Strong:</strong> Maximum protection with advanced techniques</li>
                    <li><strong>Artistic:</strong> Specialized for creative content protection</li>
                </ul>
                <p><em>Note: This provides deterrent protection. Determined actors may still attempt to use protected images.</em></p>
                </div>
                """)
            
            gr.HTML('<div class="section"><h3>🏷️ Smart Watermarking</h3></div>')
            
            add_watermark_enabled = gr.Checkbox(
                label="Add Watermark",
                value=False,
                info="Add text watermark to protect your content"
            )
            
            with gr.Row():
                watermark_text = gr.Textbox(
                    label="Watermark Text",
                    placeholder="© Your Name 2024",
                    info="Text to display as watermark"
                )
                
                watermark_position = gr.Dropdown(
                    choices=["center", "top-left", "top-right", "bottom-left", "bottom-right"],
                    value="bottom-right",
                    label="Position",
                    info="Where to place the watermark"
                )
            
            watermark_opacity = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Watermark Opacity",
                info="Transparency level (0.1 = very light, 1.0 = opaque)"
            )
            
            gr.HTML('<div class="section"><h3>✨ Auto-Enhancement</h3></div>')
            
            auto_enhance = gr.Checkbox(
                label="Auto-Enhance Images",
                value=False,
                info="Automatically improve contrast, brightness, and sharpness"
            )
            
            enhance_level = gr.Dropdown(
                choices=["subtle", "moderate", "strong"],
                value="subtle",
                label="Enhancement Level",
                info="How much enhancement to apply"
            )
            
            gr.HTML('<div class="section"><h3>📝 Batch Renaming</h3></div>')
            
            filename_pattern = gr.Textbox(
                label="Filename Pattern",
                value="{original}",
                placeholder="{original}_{number}_{date}",
                info="Pattern: {original}, {number}, {date}, {time}, {dimensions}, {width}, {height}"
            )
            
            with gr.Accordion("🔍 Filename Pattern Examples", open=False):
                gr.HTML("""
                <div style="margin: 10px 0;">
                <h4>Available Patterns:</h4>
                <ul>
                    <li><code>{original}</code> - Original filename without extension</li>
                    <li><code>{number}</code> - Sequential number (0001, 0002, etc.)</li>
                    <li><code>{date}</code> - Current date (2024-03-15)</li>
                    <li><code>{time}</code> - Current time (14-30-25)</li>
                    <li><code>{datetime}</code> - Full datetime (2024-03-15_14-30-25)</li>
                    <li><code>{width}</code> - Image width in pixels</li>
                    <li><code>{height}</code> - Image height in pixels</li>
                    <li><code>{dimensions}</code> - Width x Height (1920x1080)</li>
                </ul>
                <h4>Example Patterns:</h4>
                <ul>
                    <li><code>{original}_web</code> → photo_web.jpg</li>
                    <li><code>img_{number}_{date}</code> → img_0001_2024-03-15.jpg</li>
                    <li><code>{original}_{dimensions}</code> → photo_1920x1080.jpg</li>
                </ul>
                </div>
                """)
        
        with gr.Column(scale=1):
            gr.HTML('<div class="section"><h3>⚙️ Output Settings</h3></div>')
            
            output_format = gr.Dropdown(
                choices=["Auto", "JPEG", "PNG", "WebP"],
                value="Auto",
                label="Output Format",
                info="Auto selects best format based on content (PNG for transparency, JPEG for photos)"
            )
            
            quality = gr.Slider(
                minimum=1,
                maximum=100,
                value=85,
                step=1,
                label="JPEG/WebP Quality",
                info="Higher values = better quality but larger file size"
            )
            
            gr.HTML('<div class="section"><h3>📐 Smart Resize Settings</h3></div>')
            
            dimension_preset = gr.Dropdown(
                choices=list(DIMENSION_PRESETS.keys()),
                value="Full HD (1920x1080)",
                label="Dimension Preset",
                info="Choose a preset or select 'Custom' for manual dimensions"
            )
            
            with gr.Row():
                max_width = gr.Number(
                    label="Maximum Width (px)",
                    value=1920,
                    minimum=100,
                    maximum=8192,
                    info="Maximum width in pixels (used when preset is 'Custom')"
                )
                
                max_height = gr.Number(
                    label="Maximum Height (px)", 
                    value=1080,
                    minimum=100,
                    maximum=8192,
                    info="Maximum height in pixels (used when preset is 'Custom')"
                )
            
            gr.HTML('<div class="section"><h3>🚀 Advanced Resize Options</h3></div>')
            
            with gr.Row():
                retina_mode = gr.Checkbox(
                    label="Retina Mode (2x)",
                    value=False,
                    info="Double dimensions for high-DPI displays (retina screens)"
                )
                
                maintain_quality = gr.Checkbox(
                    label="Maintain Quality",
                    value=True,
                    info="Apply sharpening after downscaling to preserve clarity"
                )
            
            aggressive_optimization = gr.Checkbox(
                label="Aggressive Optimization",
                value=False,
                info="Apply more aggressive size reduction for large files (>5MB)"
            )
            
            gr.HTML('<div class="section"><h3>🌐 Platform Presets</h3></div>')
            
            platform_preset = gr.Dropdown(
                choices=list(PLATFORM_PRESETS.keys()),
                value="general",
                label="Platform Optimization",
                info="Optimize for specific platforms with preset dimensions and quality"
            )
            
            with gr.Accordion("📱 Platform Details", open=False):
                platform_info = gr.HTML("""
                <div style="margin: 10px 0;">
                <h4>Platform Presets:</h4>
                <ul>
                    <li><strong>General:</strong> 1920x1080, Auto format, 85% quality</li>
                    <li><strong>Instagram:</strong> 1080x1080, JPEG, 90% quality</li>
                    <li><strong>Twitter:</strong> 1200x675, JPEG, 85% quality</li>
                    <li><strong>Web Gallery:</strong> 2048x2048, Auto format, 90% quality</li>
                    <li><strong>Email:</strong> 800x600, JPEG, 75% quality</li>
                    <li><strong>Facebook:</strong> 1200x630, JPEG, 85% quality</li>
                    <li><strong>LinkedIn:</strong> 1200x627, JPEG, 85% quality</li>
                    <li><strong>Pinterest:</strong> 1000x1500, JPEG, 85% quality</li>
                </ul>
                </div>
                """)
            
            gr.HTML('<div class="section"><h3>📊 Analytics & Reporting</h3></div>')
            
            generate_report = gr.Checkbox(
                label="Generate Processing Report",
                value=False,
                info="Create detailed CSV report with analytics and file statistics"
            )
    
    gr.HTML('<div class="section"></div>')
    
    with gr.Row():
        process_btn = gr.Button(
            "🚀 Process Images",
            variant="primary",
            size="lg",
            scale=2
        )
        
        clear_btn = gr.Button(
            "🗑️ Clear",
            variant="secondary", 
            size="lg",
            scale=1
        )
    
    result_output = gr.Markdown(
        label="Processing Results",
        value="Click 'Process Images' to start batch processing..."
    )
    
    # Event handlers
    process_btn.click(
        fn=process_images,
        inputs=[
            source_dir, dest_dir, resize_enabled, strip_metadata,
            add_noise, preserve_structure, output_format, quality,
            max_width, max_height, retina_mode, aggressive_optimization,
            maintain_quality, dimension_preset, preserve_color_profile,
            preserve_copyright, add_copyright, create_metadata_log, metadata_mode,
            noise_intensity, noise_method, protection_level, preview_noise,
            add_watermark_enabled, watermark_text, watermark_position, watermark_opacity,
            auto_enhance, enhance_level, filename_pattern, platform_preset, generate_report
        ],
        outputs=result_output
    )
    
    def clear_all():
        return ("", "", True, True, False, True, "Auto", 85, 1920, 1080, 
                False, False, True, "Full HD (1920x1080)", True, False, 
                "", False, "complete", 0.05, "gaussian", "medium", False,
                False, "", "bottom-right", 0.3, False, "subtle", "{original}",
                "general", False, "Click 'Process Images' to start batch processing...")
    
    # Update dimensions when preset changes
    def update_dimensions(preset):
        if preset != "Custom":
            width, height = get_preset_dimensions(preset)
            return width, height
        return gr.update(), gr.update()
    
    dimension_preset.change(
        fn=update_dimensions,
        inputs=dimension_preset,
        outputs=[max_width, max_height]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[
            source_dir, dest_dir, resize_enabled, strip_metadata,
            add_noise, preserve_structure, output_format, quality,
            max_width, max_height, retina_mode, aggressive_optimization,
            maintain_quality, dimension_preset, preserve_color_profile,
            preserve_copyright, add_copyright, create_metadata_log, metadata_mode,
            noise_intensity, noise_method, protection_level, preview_noise,
            add_watermark_enabled, watermark_text, watermark_position, watermark_opacity,
            auto_enhance, enhance_level, filename_pattern, platform_preset, generate_report,
            result_output
        ]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 