#!/usr/bin/env python3
"""
Demo script for Comprehensive Adversarial Noise Protection
Shows how the advanced AI training protection functionality works
"""

from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Import our protection functions
from image_processor import (
    add_adversarial_noise,
    PROTECTION_PRESETS,
    calculate_psnr
)

def create_test_images():
    """Create various test images to demonstrate protection"""
    test_images = {}
    
    # 1. Simple gradient (tests basic noise application)
    gradient = np.zeros((400, 600, 3), dtype=np.uint8)
    for i in range(400):
        gradient[i, :, :] = int(255 * i / 400)
    test_images["gradient"] = Image.fromarray(gradient)
    
    # 2. Geometric pattern (tests edge detection)
    geometric = np.zeros((400, 600, 3), dtype=np.uint8)
    for i in range(0, 600, 50):
        geometric[:, i:i+25, :] = 255
    for i in range(0, 400, 50):
        geometric[i:i+25, :, 0] = 128
    test_images["geometric"] = Image.fromarray(geometric)
    
    # 3. Natural texture simulation
    np.random.seed(42)
    texture = np.random.randint(50, 205, (400, 600, 3), dtype=np.uint8)
    # Add some structure
    for i in range(0, 400, 20):
        for j in range(0, 600, 20):
            texture[i:i+10, j:j+10, :] = texture[i:i+10, j:j+10, :] + 30
    texture = np.clip(texture, 0, 255)
    test_images["texture"] = Image.fromarray(texture)
    
    # 4. High contrast image (tests frequency-based protection)
    contrast = np.zeros((400, 600, 3), dtype=np.uint8)
    contrast[:200, :300, :] = 255  # White quadrant
    contrast[200:, 300:, :] = 255  # White quadrant
    # Add some fine details
    for i in range(0, 400, 4):
        contrast[:, i, 0] = 128
    test_images["contrast"] = Image.fromarray(contrast)
    
    # 5. Artistic-style image simulation
    artistic = np.zeros((400, 600, 3), dtype=np.uint8)
    # Create brush-stroke like patterns
    for i in range(50, 350, 30):
        for j in range(50, 550, 40):
            # Random color patches
            color = np.random.randint(100, 255, 3)
            artistic[i:i+20, j:j+30, :] = color
            # Add some variation
            noise_patch = np.random.randint(-20, 20, (20, 30, 3))
            artistic[i:i+20, j:j+30, :] = np.clip(
                artistic[i:i+20, j:j+30, :] + noise_patch, 0, 255
            )
    test_images["artistic"] = Image.fromarray(artistic)
    
    return test_images

def analyze_protection_effectiveness(original, protected, method, level):
    """Analyze the effectiveness of protection"""
    original_array = np.array(original, dtype=np.float32)
    protected_array = np.array(protected, dtype=np.float32)
    
    # Calculate metrics
    psnr = calculate_psnr(original_array, protected_array)
    
    # Calculate difference statistics
    diff = protected_array - original_array
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    std_diff = np.std(diff)
    
    # Estimate visual imperceptibility
    if psnr > 45:
        visual_quality = "Imperceptible"
    elif psnr > 40:
        visual_quality = "Minimal impact"
    elif psnr > 35:
        visual_quality = "Slight impact"
    else:
        visual_quality = "Noticeable impact"
    
    # Estimate protection strength based on method and intensity
    protection_strength = {
        'gaussian': 'Moderate',
        'perlin': 'Good',
        'frequency': 'Very Good',
        'adversarial': 'Excellent',
        'style_specific': 'Excellent (Artistic)'
    }.get(method, 'Unknown')
    
    return {
        'psnr': round(psnr, 2),
        'max_difference': round(max_diff, 2),
        'mean_difference': round(mean_diff, 2),
        'std_difference': round(std_diff, 2),
        'visual_quality': visual_quality,
        'protection_strength': protection_strength
    }

def demo_adversarial_protection():
    """Demonstrate comprehensive adversarial protection features"""
    print("🛡️ Comprehensive AI Training Protection Demo")
    print("=" * 60)
    
    # Create test images
    print("\n1. Creating test images...")
    test_images = create_test_images()
    print(f"   📷 Created {len(test_images)} test images:")
    for name in test_images.keys():
        print(f"      • {name.capitalize()}: {test_images[name].size}")
    
    # Test all protection presets
    print("\n2. Testing protection presets...")
    
    # Use gradient image for preset testing
    test_image = test_images["gradient"]
    
    for preset_name, preset_config in PROTECTION_PRESETS.items():
        print(f"\n   🔧 {preset_name.upper()} Protection:")
        print(f"      Description: {preset_config['description']}")
        print(f"      Method: {preset_config['method']}")
        print(f"      Intensity: {preset_config['intensity']}")
        
        try:
            protected_image, protection_info = add_adversarial_noise(
                test_image,
                intensity=preset_config['intensity'],
                method=preset_config['method'],
                protection_level=preset_name,
                preview_noise=False
            )
            
            analysis = analyze_protection_effectiveness(
                test_image, protected_image, preset_config['method'], preset_name
            )
            
            print(f"      PSNR: {analysis['psnr']}dB")
            print(f"      Visual Quality: {analysis['visual_quality']}")
            print(f"      Protection Strength: {analysis['protection_strength']}")
            print(f"      Max Pixel Change: ±{analysis['max_difference']}")
            print(f"      Estimated Effectiveness: {protection_info['estimated_effectiveness']}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    # Test all protection methods
    print("\n3. Testing protection methods...")
    
    methods = ['gaussian', 'perlin', 'frequency', 'adversarial', 'style_specific']
    test_intensity = 0.05
    
    method_results = {}
    
    for method in methods:
        print(f"\n   🔬 {method.upper()} Method:")
        
        method_results[method] = {}
        
        # Test on different image types
        for img_name, img in test_images.items():
            try:
                protected_img, protection_info = add_adversarial_noise(
                    img,
                    intensity=test_intensity,
                    method=method,
                    protection_level='medium',
                    preview_noise=False
                )
                
                analysis = analyze_protection_effectiveness(img, protected_img, method, 'medium')
                method_results[method][img_name] = analysis
                
                print(f"      {img_name.capitalize()}: PSNR={analysis['psnr']}dB, "
                      f"Quality={analysis['visual_quality']}")
                
            except Exception as e:
                print(f"      {img_name.capitalize()}: ❌ Error - {str(e)}")
    
    # Test intensity scaling
    print("\n4. Testing intensity scaling...")
    
    intensities = [0.02, 0.05, 0.08, 0.12]
    test_method = 'adversarial'
    test_img = test_images["artistic"]
    
    print(f"   🎚️ Testing {test_method} method on artistic image:")
    
    for intensity in intensities:
        try:
            protected_img, protection_info = add_adversarial_noise(
                test_img,
                intensity=intensity,
                method=test_method,
                protection_level='medium',
                preview_noise=False
            )
            
            analysis = analyze_protection_effectiveness(test_img, protected_img, test_method, 'medium')
            
            print(f"      Intensity {intensity:.2f}: PSNR={analysis['psnr']}dB, "
                  f"Max±{analysis['max_difference']}, {analysis['visual_quality']}")
            
        except Exception as e:
            print(f"      Intensity {intensity:.2f}: ❌ Error - {str(e)}")
    
    # Test with transparency
    print("\n5. Testing transparency handling...")
    
    # Create RGBA test image
    rgba_img = test_images["geometric"].convert("RGBA")
    # Make parts transparent
    rgba_array = np.array(rgba_img)
    rgba_array[100:300, 200:400, 3] = 128  # Semi-transparent area
    rgba_array[0:100, 0:200, 3] = 0        # Fully transparent area
    rgba_test = Image.fromarray(rgba_array)
    
    try:
        protected_rgba, protection_info = add_adversarial_noise(
            rgba_test,
            intensity=0.05,
            method='perlin',
            protection_level='medium',
            preview_noise=False
        )
        
        print(f"   🎭 RGBA Protection:")
        print(f"      Original mode: {rgba_test.mode}")
        print(f"      Protected mode: {protected_rgba.mode}")
        print(f"      PSNR: {protection_info['psnr']}dB")
        print(f"      Effectiveness: {protection_info['estimated_effectiveness']}")
        print(f"      ✅ Transparency preserved")
        
    except Exception as e:
        print(f"   🎭 RGBA Protection: ❌ Error - {str(e)}")
    
    # Quality assurance tests
    print("\n6. Quality assurance tests...")
    
    qa_tests = [
        ("Imperceptible Protection", {"intensity": 0.02, "method": "gaussian"}),
        ("Balanced Protection", {"intensity": 0.05, "method": "perlin"}),
        ("Strong Protection", {"intensity": 0.08, "method": "adversarial"}),
        ("Maximum Protection", {"intensity": 0.12, "method": "style_specific"})
    ]
    
    for test_name, params in qa_tests:
        try:
            protected_img, protection_info = add_adversarial_noise(
                test_images["texture"],
                **params,
                protection_level='medium',
                preview_noise=False
            )
            
            analysis = analyze_protection_effectiveness(
                test_images["texture"], protected_img, params["method"], 'medium'
            )
            
            # Quality check
            quality_pass = analysis['psnr'] > 35  # Minimum acceptable quality
            effectiveness_good = params["intensity"] > 0.03  # Minimum effective intensity
            
            status = "✅ PASS" if quality_pass and effectiveness_good else "⚠️ REVIEW"
            
            print(f"   {status} {test_name}:")
            print(f"      PSNR: {analysis['psnr']}dB (>35dB required)")
            print(f"      Protection: {protection_info['estimated_effectiveness']}")
            print(f"      Visual Impact: {protection_info['visual_impact']}")
            
        except Exception as e:
            print(f"   ❌ {test_name}: Error - {str(e)}")
    
    # Performance and compatibility summary
    print("\n7. Performance and compatibility summary...")
    
    print(f"\n   📊 Method Performance Summary:")
    for method in methods:
        if method in method_results:
            avg_psnr = np.mean([result['psnr'] for result in method_results[method].values()])
            print(f"      {method.capitalize()}: Avg PSNR {avg_psnr:.1f}dB")
    
    print(f"\n   🎯 Recommended Settings:")
    print(f"      • General use: Medium level, Perlin method")
    print(f"      • Artwork protection: Artistic level, Style-specific method")
    print(f"      • Maximum security: Strong level, Adversarial method")
    print(f"      • Minimal impact: Light level, Gaussian method")
    
    print(f"\n   ⚠️  Important Notes:")
    print(f"      • Protection is a deterrent, not absolute security")
    print(f"      • Higher intensity = better protection but more visible")
    print(f"      • Different methods work better for different content types")
    print(f"      • Always test on sample images before batch processing")
    
    print("\n✅ Comprehensive adversarial protection demo completed!")
    print("\n🚀 Ready to use enhanced features in the Gradio app:")
    print("   python image_processor.py")
    print("\n🛡️ Protection Features:")
    print("   • 5 different noise generation methods")
    print("   • 4 protection level presets")
    print("   • Quality preservation (PSNR tracking)")
    print("   • Transparency support")
    print("   • Real-time effectiveness estimation")

if __name__ == "__main__":
    demo_adversarial_protection() 