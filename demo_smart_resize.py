#!/usr/bin/env python3
"""
Demo script for Smart Image Resizing Features
Shows how the new smart resizing functionality works
"""

from PIL import Image
import numpy as np
from pathlib import Path

# Import our smart resizing functions
from image_processor import (
    smart_resize_for_web, 
    optimize_for_web, 
    get_preset_dimensions,
    DIMENSION_PRESETS,
    calculate_file_size_mb
)

def create_demo_image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """Create a demo image for testing"""
    if mode == "RGBA":
        # Create image with transparency
        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        # Add some content
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, width-50, height-50], fill=(0, 150, 255, 180))
        draw.text((width//2-50, height//2), "DEMO IMAGE", fill=(255, 255, 255, 255))
    else:
        # Create RGB image
        img = Image.new("RGB", (width, height), (255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, width-50, height-50], fill=(0, 150, 255))
        draw.text((width//2-50, height//2), "DEMO IMAGE", fill=(255, 255, 255))
    
    return img

def demo_smart_resize():
    """Demonstrate smart resizing features"""
    print("🖼️  Smart Image Resizing Demo")
    print("=" * 50)
    
    # Create test images
    print("\n1. Creating test images...")
    large_photo = create_demo_image(4000, 3000, "RGB")  # Large photo
    large_graphic = create_demo_image(2400, 1600, "RGBA")  # Large graphic with transparency
    small_image = create_demo_image(800, 600, "RGB")  # Small image
    
    print(f"   📷 Large photo: {large_photo.size} ({calculate_file_size_mb(large_photo):.1f}MB estimated)")
    print(f"   🎨 Large graphic: {large_graphic.size} ({calculate_file_size_mb(large_graphic):.1f}MB estimated)")  
    print(f"   🖼️  Small image: {small_image.size} ({calculate_file_size_mb(small_image):.1f}MB estimated)")
    
    # Test different resize scenarios
    print("\n2. Testing smart resize scenarios...")
    
    scenarios = [
        ("Standard Web (1920x1080)", large_photo, 1920, 1080, False, False, True),
        ("Retina Mode (1920x1080)", large_photo, 1920, 1080, True, False, True),
        ("Aggressive Optimization", large_photo, 1920, 1080, False, True, True),
        ("Small Image (no resize)", small_image, 1920, 1080, False, False, True),
        ("Transparency Preserved", large_graphic, 1200, 800, False, False, True),
    ]
    
    for name, image, max_w, max_h, retina, aggressive, quality in scenarios:
        print(f"\n   🔧 {name}:")
        resized_img, resize_info = smart_resize_for_web(
            image, max_w, max_h, 
            retina_mode=retina,
            aggressive_optimization=aggressive,
            maintain_quality=quality
        )
        
        print(f"      Original: {resize_info['original_size']}")
        if resize_info['resized']:
            print(f"      Resized: {resize_info['new_size']}")
            print(f"      Scale factor: {resize_info['scale_factor']:.3f}")
            if resize_info.get('sharpening_applied'):
                print(f"      Sharpening: {resize_info['sharpening_amount']:.3f}")
        else:
            print("      No resize needed")
        
        if resize_info.get('optimization_applied'):
            print("      ⚡ Aggressive optimization applied")
    
    # Test format optimization
    print("\n3. Testing format optimization...")
    
    test_cases = [
        ("JPEG Photo", large_photo, "jpg", calculate_file_size_mb(large_photo)),
        ("PNG Graphic", large_graphic, "png", calculate_file_size_mb(large_graphic)),
        ("Large File", large_photo, "png", 8.0),  # Simulate large file
    ]
    
    for name, image, format_hint, size_mb in test_cases:
        print(f"\n   📊 {name} ({size_mb:.1f}MB):")
        optimization = optimize_for_web(image, format_hint, size_mb)
        print(f"      Recommended format: {optimization['recommended_format']}")
        print(f"      Quality setting: {optimization['quality_setting']}")
        print(f"      Reasoning: {optimization['reasoning']}")
        print(f"      Save options: {list(optimization['save_kwargs'].keys())}")
    
    # Test dimension presets
    print("\n4. Available dimension presets:")
    for preset_name, (width, height) in DIMENSION_PRESETS.items():
        print(f"   📐 {preset_name}: {width}x{height}")
    
    print("\n✅ Smart resizing demo completed!")
    print("\n🚀 Ready to use enhanced features in the Gradio app:")
    print("   python image_processor.py")

if __name__ == "__main__":
    demo_smart_resize() 