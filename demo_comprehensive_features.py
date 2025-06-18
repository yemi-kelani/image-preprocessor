#!/usr/bin/env python3

"""
Comprehensive Features Demo for Image Batch Processor

This script demonstrates all the enhanced features:
- Smart Watermarking (text and image)
- Auto-Enhancement (contrast, brightness, sharpness)
- Batch Renaming with patterns
- Platform-specific optimization presets
- Analytics and reporting
- Advanced processing combinations
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add the current directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import (
    add_watermark, auto_enhance_image, generate_filename, 
    generate_processing_report, export_report_csv,
    PLATFORM_PRESETS, DIMENSION_PRESETS,
    smart_resize_for_web, add_adversarial_noise,
    strip_comprehensive_metadata, calculate_file_size_mb
)

def create_test_images():
    """Create a set of test images for demonstration"""
    test_dir = Path("test_images_comprehensive")
    test_dir.mkdir(exist_ok=True)
    
    print("📸 Creating test images...")
    
    # Create different types of test images
    test_images = [
        ("photo_sample.jpg", (1920, 1080), "RGB", "A sample photo"),
        ("graphic_sample.png", (1200, 800), "RGBA", "A sample graphic with transparency"),
        ("portrait_sample.jpg", (1080, 1350), "RGB", "A portrait format image"),
        ("large_image.jpg", (4000, 3000), "RGB", "A large high-resolution image"),
        ("square_image.jpg", (1080, 1080), "RGB", "A square format image"),
    ]
    
    for filename, size, mode, description in test_images:
        image_path = test_dir / filename
        
        if mode == "RGBA":
            # Create image with transparency
            image = Image.new(mode, size, (100, 150, 200, 200))
            # Add some patterns
            draw = ImageDraw.Draw(image)
            for i in range(0, size[0], 100):
                draw.line([(i, 0), (i, size[1])], fill=(255, 255, 255, 100), width=2)
            for i in range(0, size[1], 100):
                draw.line([(0, i), (size[0], i)], fill=(255, 255, 255, 100), width=2)
            # Add transparent circle
            draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                        fill=(255, 0, 0, 128), outline=(255, 255, 255, 200), width=5)
        else:
            # Create RGB image
            image = Image.new(mode, size, (80, 120, 160))
            draw = ImageDraw.Draw(image)
            # Add gradient effect
            for y in range(size[1]):
                color_value = int(80 + (160 * y / size[1]))
                draw.line([(0, y), (size[0], y)], fill=(color_value, 120, 160))
            # Add some shapes
            draw.rectangle([50, 50, size[0]-50, size[1]-50], outline=(255, 255, 255), width=3)
            draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                        outline=(255, 200, 100), width=5)
        
        # Add text description
        draw.text((50, 50), description, fill=(255, 255, 255))
        draw.text((50, 80), f"Size: {size[0]}x{size[1]}", fill=(255, 255, 255))
        
        image.save(image_path, quality=95)
        print(f"  ✓ Created {filename} ({size[0]}x{size[1]}, {mode})")
    
    return test_dir

def test_watermarking():
    """Test watermarking functionality"""
    print("\n🏷️ Testing Smart Watermarking...")
    
    test_dir = create_test_images()
    output_dir = Path("output_watermarking")
    output_dir.mkdir(exist_ok=True)
    
    # Test different watermark configurations
    watermark_tests = [
        {
            "text": "© Demo 2024",
            "position": "bottom-right",
            "opacity": 0.3,
            "description": "Standard copyright watermark"
        },
        {
            "text": "SAMPLE",
            "position": "center",
            "opacity": 0.5,
            "description": "Center sample watermark"
        },
        {
            "text": "Property of Demo Studio",
            "position": "bottom-left",
            "opacity": 0.4,
            "description": "Studio watermark"
        },
        {
            "text": "CONFIDENTIAL",
            "position": "top-right",
            "opacity": 0.6,
            "description": "Confidential watermark"
        }
    ]
    
    test_image_path = test_dir / "photo_sample.jpg"
    
    with Image.open(test_image_path) as image:
        for i, test_config in enumerate(watermark_tests):
            print(f"  Testing: {test_config['description']}")
            
            watermarked_image, watermark_info = add_watermark(
                image.copy(),
                watermark_text=test_config["text"],
                position=test_config["position"],
                opacity=test_config["opacity"]
            )
            
            output_path = output_dir / f"watermarked_{i+1}_{test_config['position']}.jpg"
            watermarked_image.save(output_path, quality=90)
            
            print(f"    ✓ Watermark applied: {watermark_info['type']}")
            print(f"    ✓ Position: {watermark_info['position']}")
            print(f"    ✓ Saved: {output_path}")
    
    print(f"  📁 Watermarked images saved to: {output_dir}")

def test_auto_enhancement():
    """Test auto-enhancement functionality"""
    print("\n✨ Testing Auto-Enhancement...")
    
    test_dir = create_test_images()
    output_dir = Path("output_enhancement")
    output_dir.mkdir(exist_ok=True)
    
    # Test different enhancement levels
    enhancement_levels = ["subtle", "moderate", "strong"]
    
    test_image_path = test_dir / "photo_sample.jpg"
    
    with Image.open(test_image_path) as image:
        # Save original for comparison
        original_path = output_dir / "00_original.jpg"
        image.save(original_path, quality=90)
        print(f"  📸 Original saved: {original_path}")
        
        for level in enhancement_levels:
            print(f"  Testing enhancement level: {level}")
            
            enhanced_image, enhancement_info = auto_enhance_image(image.copy(), level)
            
            output_path = output_dir / f"enhanced_{level}.jpg"
            enhanced_image.save(output_path, quality=90)
            
            print(f"    ✓ Enhancement applied: {enhancement_info['success']}")
            print(f"    ✓ Applied adjustments: {len(enhancement_info['applied'])}")
            for adjustment in enhancement_info['applied']:
                print(f"      - {adjustment}")
            print(f"    ✓ Saved: {output_path}")
    
    print(f"  📁 Enhanced images saved to: {output_dir}")

def test_batch_renaming():
    """Test batch renaming patterns"""
    print("\n📝 Testing Batch Renaming...")
    
    test_dir = create_test_images()
    
    # Test different filename patterns
    patterns = [
        ("{original}", "Keep original names"),
        ("{original}_web", "Add web suffix"),
        ("img_{number}_{date}", "Sequential with date"),
        ("{original}_{dimensions}", "Add dimensions"),
        ("processed_{datetime}_{width}x{height}", "Full datetime with size"),
        ("{original}_enhanced_{number}", "Enhanced with counter")
    ]
    
    print("  Testing filename patterns:")
    
    for i, (pattern, description) in enumerate(patterns):
        print(f"    Pattern: {pattern} - {description}")
        
        # Test with sample image
        sample_image = Image.new("RGB", (1920, 1080), (100, 100, 100))
        
        new_filename = generate_filename(
            original_name="test_image.jpg",
            pattern=pattern,
            counter=i + 1,
            image_size=(1920, 1080),
            file_extension=".jpg"
        )
        
        print(f"      Result: test_image.jpg → {new_filename}")
    
    print("  ✓ All filename patterns tested successfully")

def test_platform_presets():
    """Test platform-specific optimization presets"""
    print("\n🌐 Testing Platform Presets...")
    
    test_dir = create_test_images()
    output_dir = Path("output_platforms")
    output_dir.mkdir(exist_ok=True)
    
    test_image_path = test_dir / "photo_sample.jpg"
    
    with Image.open(test_image_path) as image:
        print(f"  Original image: {image.size}")
        
        for platform, config in PLATFORM_PRESETS.items():
            print(f"  Testing platform: {platform}")
            print(f"    Target dimensions: {config['max_dimensions']}")
            print(f"    Format: {config['format']}")
            print(f"    Quality: {config['quality']}")
            
            # Apply platform-specific resize
            resized_image, resize_info = smart_resize_for_web(
                image.copy(),
                config['max_dimensions'][0],
                config['max_dimensions'][1],
                maintain_quality=True
            )
            
            output_path = output_dir / f"{platform}_optimized.jpg"
            resized_image.save(output_path, quality=config['quality'])
            
            print(f"    ✓ Resized: {resize_info.get('new_size', 'No resize needed')}")
            print(f"    ✓ Saved: {output_path}")
    
    print(f"  📁 Platform-optimized images saved to: {output_dir}")

def test_comprehensive_processing():
    """Test comprehensive processing with multiple features"""
    print("\n🎯 Testing Comprehensive Processing...")
    
    test_dir = create_test_images()
    output_dir = Path("output_comprehensive")
    output_dir.mkdir(exist_ok=True)
    
    # Process with multiple features enabled
    test_image_path = test_dir / "large_image.jpg"
    
    with Image.open(test_image_path) as image:
        print(f"  Original: {image.size} ({image.mode})")
        
        # Step 1: Smart resize
        print("  Step 1: Smart resize for web...")
        resized_image, resize_info = smart_resize_for_web(
            image.copy(), 1920, 1080, maintain_quality=True
        )
        print(f"    ✓ Resized: {resize_info.get('new_size', 'No change')}")
        
        # Step 2: Auto-enhance
        print("  Step 2: Auto-enhancement...")
        enhanced_image, enhancement_info = auto_enhance_image(
            resized_image, "moderate"
        )
        print(f"    ✓ Enhanced: {enhancement_info['success']}")
        
        # Step 3: Add watermark
        print("  Step 3: Add watermark...")
        watermarked_image, watermark_info = add_watermark(
            enhanced_image,
            watermark_text="© Demo Studio 2024",
            position="bottom-right",
            opacity=0.3
        )
        print(f"    ✓ Watermarked: {watermark_info['applied']}")
        
        # Step 4: Add AI protection
        print("  Step 4: Add AI protection...")
        protected_image, protection_info = add_adversarial_noise(
            watermarked_image,
            intensity=0.03,
            method="perlin",
            protection_level="medium"
        )
        print(f"    ✓ Protected: {protection_info['estimated_effectiveness']}")
        print(f"    ✓ PSNR: {protection_info['psnr']:.2f}dB")
        
        # Step 5: Strip metadata
        print("  Step 5: Strip metadata...")
        final_image, metadata_log = strip_comprehensive_metadata(
            protected_image,
            preserve_color_profile=True,
            add_copyright="© Demo Studio 2024"
        )
        print(f"    ✓ Metadata removed: {len(metadata_log['removed_items'])} items")
        
        # Step 6: Generate filename
        print("  Step 6: Generate filename...")
        new_filename = generate_filename(
            original_name="large_image.jpg",
            pattern="{original}_processed_{dimensions}",
            counter=1,
            image_size=final_image.size,
            file_extension=".jpg"
        )
        print(f"    ✓ New filename: {new_filename}")
        
        # Save final result
        output_path = output_dir / new_filename
        final_image.save(output_path, quality=90, optimize=True)
        print(f"    ✓ Final image saved: {output_path}")
        
        # Create processing summary
        processing_summary = {
            "original_size": image.size,
            "final_size": final_image.size,
            "resize_applied": resize_info.get('resized', False),
            "enhancement_applied": enhancement_info['success'],
            "watermark_applied": watermark_info['applied'],
            "protection_applied": True,
            "metadata_stripped": len(metadata_log['removed_items']) > 0,
            "final_filename": new_filename
        }
        
        print("  📊 Processing Summary:")
        for key, value in processing_summary.items():
            print(f"    {key}: {value}")

def test_analytics_reporting():
    """Test analytics and reporting functionality"""
    print("\n📊 Testing Analytics & Reporting...")
    
    # Create sample processing results
    sample_results = [
        {
            'filename': 'image1.jpg',
            'success': True,
            'original_size_mb': 2.5,
            'final_size_mb': 1.2,
            'metadata_removed': True,
            'noise_applied': True,
            'resized': True,
            'watermarked': True,
            'enhanced': True,
            'psnr': 45.2
        },
        {
            'filename': 'image2.png',
            'success': True,
            'original_size_mb': 1.8,
            'final_size_mb': 0.9,
            'metadata_removed': True,
            'noise_applied': False,
            'resized': True,
            'watermarked': False,
            'enhanced': True,
            'psnr': 0
        },
        {
            'filename': 'image3.jpg',
            'success': False,
            'original_size_mb': 0,
            'final_size_mb': 0,
            'metadata_removed': False,
            'noise_applied': False,
            'resized': False,
            'watermarked': False,
            'enhanced': False,
            'psnr': 0
        }
    ]
    
    start_time = time.time()
    time.sleep(0.1)  # Simulate processing time
    end_time = time.time()
    
    # Generate report
    report = generate_processing_report(
        source_dir="test_images",
        dest_dir="output_test",
        results=sample_results,
        start_time=start_time,
        end_time=end_time
    )
    
    print("  Generated Report Summary:")
    print(f"    Total files: {report['summary']['total_files']}")
    print(f"    Successful: {report['summary']['successful']}")
    print(f"    Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"    Processing time: {report['summary']['processing_time_seconds']:.3f}s")
    print(f"    Space saved: {report['file_sizes']['space_saved_mb']:.2f} MB ({report['file_sizes']['space_saved_percent']:.1f}%)")
    
    # Export to CSV
    report_path = "test_processing_report.csv"
    if export_report_csv(report, report_path):
        print(f"    ✓ Report exported to: {report_path}")
    else:
        print(f"    ❌ Failed to export report")

def run_comprehensive_demo():
    """Run all demo tests"""
    print("🚀 Comprehensive Image Processing Features Demo")
    print("=" * 60)
    
    try:
        # Run all tests
        test_watermarking()
        test_auto_enhancement()
        test_batch_renaming()
        test_platform_presets()
        test_comprehensive_processing()
        test_analytics_reporting()
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 60)
        print("📁 Output directories created:")
        print("  • output_watermarking/ - Watermarked images")
        print("  • output_enhancement/ - Enhanced images")
        print("  • output_platforms/ - Platform-optimized images")
        print("  • output_comprehensive/ - Comprehensive processing results")
        print("  • test_processing_report.csv - Sample analytics report")
        print("\n💡 These demonstrate all the new features in action!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_demo() 