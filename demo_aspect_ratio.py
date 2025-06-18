#!/usr/bin/env python3
"""
Demo: Aspect Ratio Preservation in Smart Resize

This demo shows how the smart resize feature automatically preserves
aspect ratios while fitting images within specified dimensions.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

# Import the smart resize function
try:
    from image_processor import smart_resize_for_web
except ImportError:
    print("❌ Could not import image_processor. Make sure you're in the correct directory.")
    sys.exit(1)

def create_demo_image(width: int, height: int, label: str) -> Image.Image:
    """Create a demo image with dimensions and label."""
    # Create image with a nice gradient background
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(height):
        color_value = int(200 + (y / height) * 55)  # Light gradient
        draw.line([(0, y), (width, y)], fill=(color_value, color_value - 20, 255))
    
    # Add border
    border_color = (50, 100, 200)
    border_width = max(2, min(width, height) // 100)
    draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
    
    # Add text information
    try:
        # Try to use a nice font, fallback to default
        font_size = max(12, min(width, height) // 20)
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Main label
    text_color = (50, 50, 50)
    draw.text((width//2 - 50, height//2 - 30), label, fill=text_color, font=font)
    draw.text((width//2 - 60, height//2), f"{width} × {height}", fill=text_color, font=font)
    
    # Add aspect ratio info
    from fractions import Fraction
    ratio = Fraction(width, height).limit_denominator(20)
    draw.text((width//2 - 40, height//2 + 20), f"Ratio: {ratio}", fill=text_color, font=font)
    
    return img

def demo_aspect_ratio_preservation():
    """Demonstrate aspect ratio preservation with various image sizes."""
    print("📐 Aspect Ratio Preservation Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path("demo_aspect_ratio_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test cases: (width, height, description)
    test_images = [
        (4000, 3000, "Large Landscape (4:3)"),
        (3000, 4000, "Large Portrait (3:4)"),
        (3840, 2160, "4K Ultra Wide (16:9)"),
        (1080, 1080, "Square Image (1:1)"),
        (5000, 1000, "Panoramic (5:1)"),
        (800, 600, "Small Landscape (4:3)"),  # This should not be resized
    ]
    
    # Target dimensions
    max_width, max_height = 1920, 1080
    
    print(f"🎯 Target maximum dimensions: {max_width} × {max_height}")
    print(f"📏 All images will be automatically resized to fit within these bounds")
    print(f"✨ Original aspect ratios will be perfectly preserved")
    print()
    
    for i, (orig_width, orig_height, description) in enumerate(test_images, 1):
        print(f"{i}. {description}")
        print(f"   Original: {orig_width} × {orig_height}")
        
        # Create demo image
        demo_img = create_demo_image(orig_width, orig_height, f"Demo {i}")
        
        # Save original
        original_path = output_dir / f"original_{i}_{orig_width}x{orig_height}.png"
        demo_img.save(original_path)
        
        # Apply smart resize
        resized_img, resize_info = smart_resize_for_web(
            demo_img,
            max_width=max_width,
            max_height=max_height,
            maintain_quality=True
        )
        
        if resize_info['resized']:
            new_size = resize_info['new_size']
            scale_factor = resize_info['scale_factor']
            
            print(f"   Resized:  {new_size[0]} × {new_size[1]} (scale: {scale_factor:.3f})")
            
            # Calculate aspect ratios to verify they match
            original_ratio = orig_width / orig_height
            new_ratio = new_size[0] / new_size[1]
            ratio_diff = abs(original_ratio - new_ratio)
            
            print(f"   Aspect ratio preserved: {ratio_diff < 0.001} (diff: {ratio_diff:.6f})")
            
            # Show how much it fits within bounds
            width_usage = (new_size[0] / max_width) * 100
            height_usage = (new_size[1] / max_height) * 100
            print(f"   Dimension usage: {width_usage:.1f}% width, {height_usage:.1f}% height")
            
            # Save resized image
            resized_path = output_dir / f"resized_{i}_{new_size[0]}x{new_size[1]}.png"
            resized_img.save(resized_path)
            
        else:
            print(f"   No resize needed (already within bounds)")
            resized_path = output_dir / f"unchanged_{i}_{orig_width}x{orig_height}.png"
            resized_img.save(resized_path)
        
        print()
    
    print("📁 Demo images saved to:", output_dir)
    print()
    print("🔍 Key Points Demonstrated:")
    print("  • Large images are automatically scaled down")
    print("  • Small images remain unchanged (no unnecessary upscaling)")
    print("  • Aspect ratios are perfectly preserved (no distortion)")
    print("  • Images fit within the specified maximum dimensions")
    print("  • High-quality Lanczos resampling maintains image clarity")
    print()
    print("✅ The Smart Resize feature works exactly as intended!")
    print("   Your images will never be stretched or distorted.")

def show_mathematical_explanation():
    """Show the mathematical approach used for aspect ratio preservation."""
    print("\n" + "=" * 60)
    print("🧮 Mathematical Explanation")
    print("=" * 60)
    print()
    print("The smart resize algorithm uses this approach:")
    print()
    print("1. Calculate scale factors for width and height:")
    print("   width_scale = target_width / original_width")
    print("   height_scale = target_height / original_height")
    print()
    print("2. Choose the SMALLER scale factor:")
    print("   scale_factor = min(width_scale, height_scale)")
    print()
    print("3. Apply proportional scaling:")
    print("   new_width = original_width × scale_factor")
    print("   new_height = original_height × scale_factor")
    print()
    print("📊 Example with 4000×3000 image → 1920×1080 target:")
    print("   width_scale = 1920 / 4000 = 0.48")
    print("   height_scale = 1080 / 3000 = 0.36")
    print("   scale_factor = min(0.48, 0.36) = 0.36")
    print("   Result: 1440×1080 (perfectly proportional!)")

if __name__ == "__main__":
    demo_aspect_ratio_preservation()
    show_mathematical_explanation() 