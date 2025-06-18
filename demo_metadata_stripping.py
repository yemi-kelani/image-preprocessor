#!/usr/bin/env python3
"""
Demo script for Comprehensive Metadata Stripping Features
Shows how the advanced metadata handling functionality works
"""

from PIL import Image
from PIL.ExifTags import TAGS
import piexif
from pathlib import Path
import json

# Import our metadata functions
from image_processor import (
    detect_metadata,
    strip_comprehensive_metadata,
    sanitize_metadata
)

def create_demo_image_with_metadata() -> Image.Image:
    """Create a demo image with comprehensive metadata"""
    img = Image.new("RGB", (1200, 800), (100, 150, 200))
    
    # Create comprehensive EXIF data
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "Demo Camera Corp",
            piexif.ImageIFD.Model: "DemoCamera X1",
            piexif.ImageIFD.Software: "DemoApp v1.0",
            piexif.ImageIFD.DateTime: "2024:01:15 10:30:45",
            piexif.ImageIFD.Artist: "Demo Photographer",
            piexif.ImageIFD.Copyright: "© Demo Corp 2024",
            piexif.ImageIFD.ImageDescription: "Demo image with metadata",
            piexif.ImageIFD.XPComment: "Personal photo".encode('utf-16le'),
            piexif.ImageIFD.XPAuthor: "John Doe".encode('utf-16le'),
            piexif.ImageIFD.XPKeywords: "demo;test;metadata".encode('utf-16le')
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2024:01:15 10:30:45",
            piexif.ExifIFD.DateTimeDigitized: "2024:01:15 10:30:46",
            piexif.ExifIFD.ISOSpeedRatings: 400,
            piexif.ExifIFD.FNumber: (280, 100),  # f/2.8
            piexif.ExifIFD.ExposureTime: (1, 60),  # 1/60s
            piexif.ExifIFD.FocalLength: (50, 1),  # 50mm
            piexif.ExifIFD.LensMake: "Demo Lens Corp",
            piexif.ExifIFD.LensModel: "Demo 50mm f/2.8",
            piexif.ExifIFD.UserComment: "Beautiful sunset photo".encode('ascii')
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((37, 1), (46, 1), (0, 1)),  # 37.7749° N (San Francisco)
            piexif.GPSIFD.GPSLongitudeRef: "W", 
            piexif.GPSIFD.GPSLongitude: ((122, 1), (25, 1), (0, 1)),  # 122.4194° W
            piexif.GPSIFD.GPSAltitude: (50, 1),  # 50 meters
            piexif.GPSIFD.GPSDateStamp: "2024:01:15"
        },
        "1st": {},
        "thumbnail": None
    }
    
    # Add EXIF to image
    exif_bytes = piexif.dump(exif_dict)
    img.info["exif"] = exif_bytes
    
    # Add ICC color profile (dummy profile)
    img.info["icc_profile"] = b"dummy_icc_profile_data_" * 100
    
    # Add XMP metadata
    img.info["xmp"] = """<?xml version="1.0" encoding="UTF-8"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="">
                <dc:creator>Demo Creator</dc:creator>
                <dc:subject>demo, test, metadata</dc:subject>
                <dc:description>Demo image for metadata testing</dc:description>
                <xmp:CreateDate>2024-01-15T10:30:45</xmp:CreateDate>
                <xmp:CreatorTool>Demo Software v1.0</xmp:CreatorTool>
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>"""
    
    return img

def analyze_exif_data(exif_dict):
    """Analyze and display EXIF data in human-readable format"""
    info = []
    
    for ifd_name, ifd_dict in exif_dict.items():
        if not ifd_dict or ifd_name == "thumbnail":
            continue
            
        info.append(f"\n  {ifd_name.upper()} Data:")
        for tag, value in ifd_dict.items():
            try:
                if ifd_name == "GPS":
                    tag_name = piexif.GPSIFD.get(tag, f"Unknown_{tag}")
                elif ifd_name == "Exif":
                    tag_name = piexif.ExifIFD.get(tag, f"Unknown_{tag}")
                else:
                    tag_name = piexif.ImageIFD.get(tag, f"Unknown_{tag}")
                
                # Format specific values
                if isinstance(value, bytes):
                    if len(value) > 50:
                        display_value = f"<{len(value)} bytes>"
                    else:
                        try:
                            display_value = value.decode('utf-8', errors='ignore')
                        except:
                            display_value = f"<{len(value)} bytes>"
                elif isinstance(value, tuple) and len(value) == 2:
                    display_value = f"{value[0]}/{value[1]} ({value[0]/value[1]:.2f})"
                else:
                    display_value = str(value)
                
                info.append(f"    {tag_name}: {display_value}")
            except Exception as e:
                info.append(f"    Tag {tag}: <error reading value>")
    
    return info

def demo_metadata_stripping():
    """Demonstrate comprehensive metadata stripping features"""
    print("🔒 Comprehensive Metadata Stripping Demo")
    print("=" * 60)
    
    # Create test image with metadata
    print("\n1. Creating demo image with comprehensive metadata...")
    demo_image = create_demo_image_with_metadata()
    print(f"   📷 Image created: {demo_image.size}")
    print(f"   📊 Image info keys: {list(demo_image.info.keys())}")
    
    # Detect metadata
    print("\n2. Detecting metadata...")
    metadata_info = detect_metadata(demo_image)
    
    print(f"   ✅ Has EXIF: {metadata_info['has_exif']}")
    print(f"   ✅ Has ICC Profile: {metadata_info['has_icc_profile']}")
    print(f"   ✅ Has XMP: {metadata_info['has_xmp']}")
    print(f"   ✅ Total metadata size: {metadata_info['total_metadata_size']} bytes")
    
    if metadata_info['sensitive_data_found']:
        print(f"   ⚠️  Sensitive data detected:")
        for item in metadata_info['sensitive_data_found'][:5]:  # Show first 5
            print(f"      • {item}")
        if len(metadata_info['sensitive_data_found']) > 5:
            print(f"      • ... and {len(metadata_info['sensitive_data_found']) - 5} more")
    
    # Show detailed EXIF data
    if metadata_info['has_exif']:
        print("\n3. Detailed EXIF analysis:")
        exif_analysis = analyze_exif_data(metadata_info['exif_data'])
        for line in exif_analysis[:15]:  # Show first 15 lines
            print(line)
        if len(exif_analysis) > 15:
            print(f"   ... and {len(exif_analysis) - 15} more lines")
    
    # Test comprehensive metadata removal
    print("\n4. Testing comprehensive metadata removal...")
    
    scenarios = [
        ("Complete Removal", {
            "preserve_color_profile": False,
            "preserve_copyright": False,
            "add_copyright": None
        }),
        ("Preserve Color Profile", {
            "preserve_color_profile": True,
            "preserve_copyright": False,
            "add_copyright": None
        }),
        ("Preserve Copyright", {
            "preserve_color_profile": True,
            "preserve_copyright": True,
            "add_copyright": None
        }),
        ("Add Custom Copyright", {
            "preserve_color_profile": True,
            "preserve_copyright": False,
            "add_copyright": "© Enhanced Demo 2024"
        })
    ]
    
    for scenario_name, params in scenarios:
        print(f"\n   🔧 {scenario_name}:")
        clean_image, removal_log = strip_comprehensive_metadata(
            demo_image,
            **params,
            create_log=True
        )
        
        print(f"      Removed items: {len(removal_log['removed_items'])}")
        print(f"      Preserved items: {len(removal_log['preserved_items'])}")
        print(f"      Added items: {len(removal_log['added_items'])}")
        print(f"      Size reduction: {removal_log['size_reduction']} bytes")
        print(f"      Final image info: {list(clean_image.info.keys())}")
        
        # Show some details
        if removal_log['removed_items'][:3]:
            print(f"      Sample removed: {removal_log['removed_items'][:3]}")
        if removal_log['preserved_items']:
            print(f"      Preserved: {removal_log['preserved_items']}")
        if removal_log['added_items']:
            print(f"      Added: {removal_log['added_items']}")
    
    # Test selective sanitization
    print("\n5. Testing selective metadata sanitization...")
    
    sanitization_options = [
        ("Remove GPS Only", {
            "remove_gps": True,
            "remove_timestamps": False,
            "remove_camera_info": False,
            "remove_software_info": False,
            "preserve_copyright": True
        }),
        ("Remove Sensitive Data", {
            "remove_gps": True,
            "remove_timestamps": True,
            "remove_camera_info": True,
            "remove_software_info": True,
            "preserve_copyright": True
        }),
        ("Minimal Sanitization", {
            "remove_gps": True,
            "remove_timestamps": False,
            "remove_camera_info": False,
            "remove_software_info": False,
            "preserve_copyright": True
        })
    ]
    
    for option_name, params in sanitization_options:
        print(f"\n   🧹 {option_name}:")
        sanitized_image, sanitization_log = sanitize_metadata(demo_image, **params)
        
        print(f"      Removed: {len(set(sanitization_log['removed_items']))}")
        print(f"      Preserved: {len(sanitization_log['preserved_items'])}")
        print(f"      Final image info: {list(sanitized_image.info.keys())}")
        
        # Show unique removed items
        unique_removed = list(set(sanitization_log['removed_items']))
        if unique_removed:
            print(f"      Removed types: {unique_removed}")
    
    print("\n6. Privacy and security analysis...")
    
    # Test different scenarios for privacy
    privacy_tests = [
        ("Standard Privacy", demo_image, {"preserve_color_profile": True}),
        ("Maximum Privacy", demo_image, {"preserve_color_profile": False}),
        ("Business Use", demo_image, {"preserve_color_profile": True, "preserve_copyright": True})
    ]
    
    for test_name, test_image, params in privacy_tests:
        clean_img, log = strip_comprehensive_metadata(test_image, **params)
        remaining_metadata = detect_metadata(clean_img)
        
        print(f"\n   🔐 {test_name}:")
        print(f"      Remaining EXIF: {remaining_metadata['has_exif']}")
        print(f"      Remaining sensitive data: {len(remaining_metadata['sensitive_data_found'])}")
        print(f"      Privacy score: {'🟢 High' if len(remaining_metadata['sensitive_data_found']) == 0 else '🟡 Medium' if len(remaining_metadata['sensitive_data_found']) < 3 else '🔴 Low'}")
    
    print("\n✅ Comprehensive metadata stripping demo completed!")
    print("\n🚀 Ready to use enhanced features in the Gradio app:")
    print("   python image_processor.py")
    print("\n🔒 Privacy Features:")
    print("   • Complete metadata removal")
    print("   • Selective sanitization") 
    print("   • Color profile preservation")
    print("   • Copyright management")
    print("   • Detailed logging")

if __name__ == "__main__":
    demo_metadata_stripping() 