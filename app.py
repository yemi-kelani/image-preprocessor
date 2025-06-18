"""
Professional Image Batch Processor - Main Application

A comprehensive, production-ready image processing application with:
- Smart resizing and optimization
- Metadata privacy protection  
- AI training protection
- Watermarking and enhancement
- Platform-specific presets
- Detailed analytics and reporting
"""

import gradio as gr
import os
import sys
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Import from the comprehensive image processor
from image_processor import (
    find_images, process_single_image, 
    PLATFORM_PRESETS, DIMENSION_PRESETS,
    generate_processing_report, export_report_csv
)

class ImageBatchProcessor:
    """Main application class for professional image batch processing."""
    
    def __init__(self):
        self.setup_logging()
        self.processing_state = {
            'is_running': False,
            'can_cancel': False,
            'current_file': '',
            'progress': 0,
            'total_files': 0,
            'start_time': 0,
            'results': []
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('image_processor.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Image Batch Processor initialized")
    
    def validate_inputs(self, source_dir: str, dest_dir: str) -> Tuple[bool, str]:
        """Validate user inputs before processing."""
        if not source_dir or not source_dir.strip():
            return False, "❌ Please specify a source directory"
        
        if not dest_dir or not dest_dir.strip():
            return False, "❌ Please specify a destination directory"
        
        source_path = Path(source_dir.strip())
        if not source_path.exists():
            return False, f"❌ Source directory does not exist: {source_dir}"
        
        if not source_path.is_dir():
            return False, f"❌ Source path is not a directory: {source_dir}"
        
        try:
            dest_path = Path(dest_dir.strip())
            dest_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"❌ Cannot create destination directory: {e}"
        
        # Check if directories are the same
        try:
            if source_path.samefile(dest_path):
                return False, "❌ Source and destination directories cannot be the same"
        except FileNotFoundError:
            pass  # Destination doesn't exist yet, which is fine
        
        return True, "✅ Directories are valid"
    
    def scan_images(self, source_dir: str) -> Tuple[List[Path], Dict]:
        """Scan source directory for images and return statistics."""
        try:
            image_files = find_images(source_dir)
            
            if not image_files:
                return [], {"error": "No supported image files found"}
            
            # Calculate statistics
            total_size = sum(f.stat().st_size for f in image_files)
            format_counts = {}
            for f in image_files:
                ext = f.suffix.lower()
                format_counts[ext] = format_counts.get(ext, 0) + 1
            
            stats = {
                "total_files": len(image_files),
                "total_size_mb": total_size / (1024 * 1024),
                "formats": format_counts,
                "avg_size_mb": (total_size / len(image_files)) / (1024 * 1024)
            }
            
            return image_files, stats
            
        except Exception as e:
            self.logger.error(f"Error scanning images: {e}")
            return [], {"error": str(e)}
    
    def estimate_processing_time(self, file_count: int, avg_size_mb: float, features: List[str]) -> str:
        """Estimate processing time based on enabled features."""
        base_time = 0.5  # seconds per file
        feature_multipliers = {
            'resize': 1.2,
            'metadata': 1.1,
            'watermark': 1.15,
            'enhancement': 1.3,
            'protection': 1.8
        }
        
        multiplier = 1.0
        for feature in features:
            multiplier *= feature_multipliers.get(feature, 1.0)
        
        # Size factor
        if avg_size_mb > 10:
            multiplier *= 1.5
        elif avg_size_mb > 5:
            multiplier *= 1.2
        
        total_seconds = file_count * base_time * multiplier
        
        if total_seconds < 60:
            return f"~{total_seconds:.0f} seconds"
        elif total_seconds < 3600:
            return f"~{total_seconds/60:.1f} minutes"
        else:
            return f"~{total_seconds/3600:.1f} hours"
    
    def process_batch(self, 
                     source_dir: str,
                     dest_dir: str,
                     # Processing options
                     resize_enabled: bool,
                     strip_metadata: bool,
                     add_noise: bool,
                     preserve_structure: bool,
                     output_format: str,
                     quality: int,
                     max_width: int,
                     max_height: int,
                     # Advanced options
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
                     progress=gr.Progress()) -> str:
        """
        Main batch processing function with comprehensive features.
        """
        
        # Validate inputs
        is_valid, message = self.validate_inputs(source_dir, dest_dir)
        if not is_valid:
            return message
        
        try:
            # Find images
            progress(0, desc="📁 Scanning for images...")
            image_files, scan_stats = self.scan_images(source_dir)
            
            if not image_files:
                error_msg = scan_stats.get('error', 'No images found')
                return f"❌ {error_msg}"
            
            # Apply platform preset if specified
            if platform_preset != "general" and platform_preset in PLATFORM_PRESETS:
                preset_config = PLATFORM_PRESETS[platform_preset]
                if dimension_preset == "Custom":
                    max_width, max_height = preset_config["max_dimensions"]
                if output_format.upper() == "AUTO":
                    output_format = preset_config["format"]
                    if "quality" in preset_config:
                        quality = preset_config["quality"]
            
            # Use preset dimensions if not custom
            if dimension_preset != "Custom" and dimension_preset in DIMENSION_PRESETS:
                max_width, max_height = DIMENSION_PRESETS[dimension_preset]
            
            # Process images
            processed_count = 0
            error_count = 0
            error_messages = []
            detailed_results = []
            
            start_time = time.time()
            
            for i, image_file in enumerate(image_files):
                progress_pct = (i + 1) / len(image_files)
                progress(progress_pct, desc=f"🖼️ Processing {image_file.name}...")
                
                # Clean copyright parameter
                copyright_notice = add_copyright.strip() if add_copyright and add_copyright.strip() else None
                
                try:
                    success, message, result_info = process_single_image(
                        image_file, Path(source_dir), Path(dest_dir),
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
                        
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing {image_file.name}: {str(e)}"
                    error_messages.append(error_msg)
                    self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    
                    # Add error result
                    detailed_results.append({
                        'filename': image_file.name,
                        'success': False,
                        'message': error_msg,
                        'original_size_mb': 0,
                        'final_size_mb': 0,
                        'metadata_removed': False,
                        'noise_applied': False,
                        'resized': False,
                        'watermarked': False,
                        'enhanced': False,
                        'psnr': 0
                    })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Generate comprehensive report
            if generate_report and detailed_results:
                try:
                    report = generate_processing_report(source_dir, dest_dir, detailed_results, start_time, end_time)
                    report_path = Path(dest_dir) / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    if export_report_csv(report, str(report_path)):
                        report_info = f"📄 **Detailed Report:** `{report_path}`\n\n"
                    else:
                        report_info = "⚠️ Report generation failed\n\n"
                except Exception as e:
                    report_info = f"⚠️ Report generation error: {str(e)}\n\n"
            else:
                report_info = ""
            
            # Calculate analytics
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
            
            # Build result message
            result_lines = [
                "🎉 **Processing Complete!**",
                "",
                report_info,
                f"📊 **Summary Statistics:**",
                f"• **Files processed:** {processed_count}/{len(image_files)} ({processed_count/len(image_files)*100:.1f}%)",
                f"• **Processing time:** {processing_time:.1f}s ({processing_time/len(image_files):.2f}s per file)",
                f"• **Errors:** {error_count}",
                ""
            ]
            
            if successful_results:
                result_lines.extend([
                    f"💾 **File Size Optimization:**",
                    f"• **Original total:** {total_original_mb:.1f} MB",
                    f"• **Final total:** {total_final_mb:.1f} MB",
                    f"• **Space saved:** {space_saved_mb:.1f} MB ({space_saved_percent:.1f}%)",
                    ""
                ])
                
                if any([metadata_stripped_count, noise_applied_count, resized_count, watermarked_count, enhanced_count]):
                    result_lines.extend([
                        f"🔧 **Features Applied:**"
                    ])
                    if resized_count > 0:
                        result_lines.append(f"• **Smart resize:** {resized_count} images")
                    if metadata_stripped_count > 0:
                        result_lines.append(f"• **Metadata removal:** {metadata_stripped_count} images")
                    if noise_applied_count > 0:
                        result_lines.append(f"• **AI protection:** {noise_applied_count} images")
                    if watermarked_count > 0:
                        result_lines.append(f"• **Watermarks:** {watermarked_count} images")
                    if enhanced_count > 0:
                        result_lines.append(f"• **Auto-enhancement:** {enhanced_count} images")
                    result_lines.append("")
            
            result_lines.extend([
                f"📁 **Output location:** `{dest_dir}`",
                ""
            ])
            
            if error_count > 0:
                result_lines.extend([
                    f"❌ **Errors ({error_count}):**"
                ])
                for error_msg in error_messages[:5]:  # Show first 5 errors
                    result_lines.append(f"• {error_msg}")
                if len(error_messages) > 5:
                    result_lines.append(f"• ... and {len(error_messages) - 5} more errors")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}\n{traceback.format_exc()}")
            return f"❌ **Processing failed:** {str(e)}"
    
    def create_ui(self) -> gr.Blocks:
        """Create the comprehensive Gradio interface."""
        
        # Theme-aware CSS that works in both light and dark modes
        custom_css = """
        .container { 
            max-width: 1400px; 
            margin: auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 2rem; 
        }
        .section { 
            margin: 1.5rem 0; 
            padding: 1rem; 
            border: 1px solid var(--border-color-primary); 
            border-radius: 8px; 
            background: var(--background-fill-secondary);
            transition: all 0.3s ease;
        }
        .feature-section { 
            margin: 1rem 0; 
            padding: 0.8rem; 
            border-left: 4px solid var(--color-accent); 
            background: var(--background-fill-primary);
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        .status-box { 
            padding: 1rem; 
            border-radius: 8px; 
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
            color: var(--body-text-color);
            transition: all 0.3s ease;
        }
        .error-box { 
            padding: 1rem; 
            border-radius: 8px; 
            background: var(--background-fill-primary);
            border: 1px solid #ef4444;
            color: #ef4444;
            transition: all 0.3s ease;
        }
        .success-box { 
            padding: 1rem; 
            border-radius: 8px; 
            background: var(--background-fill-primary);
            border: 1px solid #22c55e;
            color: #22c55e;
            transition: all 0.3s ease;
        }
        
        /* Ensure text readability in both themes */
        .section h3,
        .feature-section h3 {
            color: var(--body-text-color);
            margin-top: 0;
        }
        
        /* Responsive design improvements */
        @media (max-width: 768px) {
            .container {
                max-width: 100%;
                padding: 0 1rem;
            }
            .section,
            .feature-section {
                margin: 1rem 0;
                padding: 0.75rem;
            }
        }
        
        /* Enhance button styling for better theme compatibility */
        .gradio-button {
            transition: all 0.2s ease !important;
        }
        
        /* Improve accordion styling */
        .gradio-accordion {
            border: 1px solid var(--border-color-primary) !important;
            border-radius: 8px !important;
        }
        """
        
        with gr.Blocks(
            title="Professional Image Batch Processor",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as app:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1 style="color: var(--body-text-color); margin-bottom: 0.5rem;">🎨 Professional Image Batch Processor</h1>
                <p style="font-size: 1.1em; color: var(--body-text-color-subdued); margin-top: 0;">
                    Production-ready batch processing with smart optimization, privacy protection, and AI training deterrence
                </p>
            </div>
            """)
            
            # Main interface with tabs
            with gr.Tabs() as main_tabs:
                
                # Process Tab
                with gr.Tab("🚀 Process", id="process"):
                    with gr.Row():
                        # Left Column - Directory and Basic Settings
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section"><h3 style="color: var(--body-text-color);">📁 Directory Settings</h3></div>')
                            
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
                            
                            # Directory validation
                            dir_status = gr.HTML("<div class='status-box'>📋 Enter directories to validate</div>")
                            
                            # Platform presets
                            gr.HTML('<div class="feature-section"><h3 style="color: var(--body-text-color);">🌐 Platform Optimization</h3></div>')
                            
                            platform_preset = gr.Dropdown(
                                choices=list(PLATFORM_PRESETS.keys()),
                                value="general",
                                label="Platform Preset",
                                info="Pre-configured settings for specific platforms"
                            )
                            
                            # Quick scan button
                            scan_btn = gr.Button("🔍 Scan Images", variant="secondary")
                            scan_results = gr.HTML("")
                            
                            # Processing features
                            gr.HTML('<div class="feature-section"><h3 style="color: var(--body-text-color);">🔧 Processing Features</h3></div>')
                            
                            with gr.Row():
                                resize_enabled = gr.Checkbox(
                                    label="Smart Resize (Aspect Ratio Preserved)",
                                    value=True,
                                    info="Intelligent web optimization with automatic aspect ratio preservation"
                                )
                                
                                strip_metadata = gr.Checkbox(
                                    label="Strip Metadata",
                                    value=True,
                                    info="Privacy protection"
                                )
                            
                            with gr.Row():
                                add_noise = gr.Checkbox(
                                    label="AI Protection",
                                    value=False,
                                    info="Deterrent against AI training"
                                )
                                
                                add_watermark_enabled = gr.Checkbox(
                                    label="Add Watermark",
                                    value=False,
                                    info="Brand protection"
                                )
                            
                            with gr.Row():
                                auto_enhance = gr.Checkbox(
                                    label="Auto-Enhance",
                                    value=False,
                                    info="Improve image quality"
                                )
                                
                                generate_report = gr.Checkbox(
                                    label="Generate Report",
                                    value=True,
                                    info="Detailed analytics CSV"
                                )
                        
                        # Right Column - Advanced Settings
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section"><h3 style="color: var(--body-text-color);">⚙️ Advanced Settings</h3></div>')
                            
                            # Output settings
                            with gr.Accordion("📤 Output Settings", open=True):
                                output_format = gr.Dropdown(
                                    choices=["Auto", "JPEG", "PNG", "WebP"],
                                    value="Auto",
                                    label="Output Format"
                                )
                                
                                quality = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=90,
                                    step=1,
                                    label="Quality"
                                )
                                
                                preserve_structure = gr.Checkbox(
                                    label="Preserve Folder Structure",
                                    value=True
                                )
                            
                            # Resize settings
                            with gr.Accordion("📐 Smart Resize Settings (Aspect Ratio Preserved)", open=True):
                                gr.HTML("""
                                <div style="background: var(--background-fill-primary); padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid var(--color-accent);">
                                    <strong>🔄 Automatic Aspect Ratio Preservation</strong><br/>
                                    <small style="color: var(--body-text-color-subdued);">Images are automatically resized to fit within the specified dimensions while maintaining their original proportions. No distortion or stretching occurs.</small>
                                </div>
                                """)
                                
                                dimension_preset = gr.Dropdown(
                                    choices=list(DIMENSION_PRESETS.keys()),
                                    value="Full HD (1920x1080)",
                                    label="Dimension Preset",
                                    info="Common size presets for different platforms and uses"
                                )
                                
                                with gr.Row():
                                    max_width = gr.Number(
                                        label="Max Width (px)",
                                        value=1920,
                                        minimum=100,
                                        maximum=8192,
                                        info="Maximum width - images wider than this will be resized proportionally"
                                    )
                                    
                                    max_height = gr.Number(
                                        label="Max Height (px)", 
                                        value=1080,
                                        minimum=100,
                                        maximum=8192,
                                        info="Maximum height - images taller than this will be resized proportionally"
                                    )
                                
                                gr.HTML("""
                                <div style="background: var(--background-fill-secondary); padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0;">
                                    <small><strong>📏 How it works:</strong> Images larger than the specified dimensions are automatically scaled down to fit within the bounds while preserving their original aspect ratio. Smaller images remain unchanged unless Retina Mode is enabled.</small>
                                </div>
                                """)
                                
                                with gr.Row():
                                    retina_mode = gr.Checkbox(
                                        label="Retina Mode (2x)",
                                        value=False,
                                        info="Double the target dimensions for high-DPI displays"
                                    )
                                    
                                    maintain_quality = gr.Checkbox(
                                        label="Maintain Quality",
                                        value=True,
                                        info="Apply intelligent sharpening after resize to preserve image clarity"
                                    )
                                
                                aggressive_optimization = gr.Checkbox(
                                    label="Aggressive Optimization",
                                    value=False,
                                    info="Extra compression for large files (>5MB) - reduces dimensions by 20%"
                                )
                            
                            # Metadata settings
                            with gr.Accordion("🔒 Privacy Settings", open=False):
                                metadata_mode = gr.Dropdown(
                                    choices=["complete", "sanitize"],
                                    value="complete",
                                    label="Metadata Removal Mode"
                                )
                                
                                with gr.Row():
                                    preserve_color_profile = gr.Checkbox(
                                        label="Preserve Color Profile",
                                        value=True
                                    )
                                    
                                    preserve_copyright = gr.Checkbox(
                                        label="Preserve Copyright",
                                        value=False
                                    )
                                
                                add_copyright = gr.Textbox(
                                    label="Add Copyright Notice",
                                    placeholder="© Your Name 2024"
                                )
                            
                            # AI Protection settings
                            with gr.Accordion("🛡️ AI Protection", open=False):
                                protection_level = gr.Dropdown(
                                    choices=["light", "medium", "strong", "artistic"],
                                    value="medium",
                                    label="Protection Level"
                                )
                                
                                noise_method = gr.Dropdown(
                                    choices=["gaussian", "perlin", "frequency", "adversarial", "style_specific"],
                                    value="perlin",
                                    label="Protection Method"
                                )
                                
                                noise_intensity = gr.Slider(
                                    minimum=0.01,
                                    maximum=0.15,
                                    value=0.05,
                                    step=0.01,
                                    label="Intensity"
                                )
                            
                            # Watermark settings
                            with gr.Accordion("🏷️ Watermark Settings", open=False):
                                watermark_text = gr.Textbox(
                                    label="Watermark Text",
                                    placeholder="© Your Brand 2024"
                                )
                                
                                with gr.Row():
                                    watermark_position = gr.Dropdown(
                                        choices=["center", "top-left", "top-right", "bottom-left", "bottom-right"],
                                        value="bottom-right",
                                        label="Position"
                                    )
                                    
                                    watermark_opacity = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.0,
                                        value=0.3,
                                        step=0.1,
                                        label="Opacity"
                                    )
                            
                            # Enhancement settings
                            with gr.Accordion("✨ Enhancement Settings", open=False):
                                enhance_level = gr.Dropdown(
                                    choices=["subtle", "moderate", "strong"],
                                    value="subtle",
                                    label="Enhancement Level"
                                )
                            
                            # Naming settings
                            with gr.Accordion("📝 File Naming", open=False):
                                filename_pattern = gr.Textbox(
                                    label="Filename Pattern",
                                    value="{original}",
                                    placeholder="{original}_{dimensions}"
                                )
                                
                                gr.HTML("""
                                <small><strong>Available patterns:</strong> {original}, {number}, {date}, {time}, {dimensions}, {width}, {height}</small>
                                """)
                            
                            # Hidden components for event handlers
                            create_metadata_log = gr.Checkbox(value=False, visible=False)
                            preview_noise = gr.Checkbox(value=False, visible=False)
                    
                    # Processing controls
                    gr.HTML('<div class="section"></div>')
                    
                    with gr.Row():
                        process_btn = gr.Button(
                            "🚀 Process Images",
                            variant="primary",
                            size="lg",
                            scale=3
                        )
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            variant="secondary", 
                            size="lg",
                            scale=1
                        )
                    
                    # Results
                    result_output = gr.Markdown(
                        label="Processing Results",
                        value="🎯 **Ready to process images**\n\nChoose your source and destination directories, configure processing options, and click 'Process Images' to start."
                    )
                
                # Help Tab
                with gr.Tab("❓ Help", id="help"):
                    gr.HTML("""
                    <div class="section">
                        <h3>📚 Help & Documentation</h3>
                        
                        <h4>🎯 Quick Start</h4>
                        <ol>
                            <li><strong>Set Directories:</strong> Enter source folder with images and destination for processed files</li>
                            <li><strong>Choose Platform:</strong> Select a platform preset for optimized settings</li>
                            <li><strong>Enable Features:</strong> Check desired processing options (resize, metadata removal, etc.)</li>
                            <li><strong>Process:</strong> Click "Process Images" and monitor progress</li>
                        </ol>
                        
                        <h4>🔧 Feature Overview</h4>
                        <ul>
                            <li><strong>Smart Resize:</strong> Automatically resize images to fit within specified dimensions while preserving aspect ratio. No distortion or stretching - images maintain their original proportions</li>
                            <li><strong>Strip Metadata:</strong> Remove EXIF, GPS, and other privacy-sensitive data</li>
                            <li><strong>AI Protection:</strong> Add imperceptible noise to deter AI training</li>
                            <li><strong>Watermarking:</strong> Add text watermarks for brand protection</li>
                            <li><strong>Auto-Enhancement:</strong> Improve contrast, brightness, and sharpness</li>
                            <li><strong>Platform Presets:</strong> Optimized settings for Instagram, Twitter, etc.</li>
                        </ul>
                        
                        <h4>📐 Aspect Ratio Preservation Details</h4>
                        <ul>
                            <li><strong>How it works:</strong> The smart resize feature automatically calculates the optimal size to fit your images within the specified maximum dimensions</li>
                            <li><strong>No distortion:</strong> Images are scaled proportionally - tall images remain tall, wide images remain wide</li>
                            <li><strong>Quality preservation:</strong> High-quality Lanczos resampling with optional intelligent sharpening</li>
                            <li><strong>Smart scaling:</strong> Only resizes images that are larger than the target dimensions</li>
                            <li><strong>Example:</strong> A 4000x3000 image with max dimensions 1920x1080 becomes 1440x1080 (maintains 4:3 ratio)</li>
                        </ul>
                        
                        <h4>📊 Platform Presets</h4>
                        <ul>
                            <li><strong>Instagram:</strong> 1080x1080, JPEG, 90% quality</li>
                            <li><strong>Twitter:</strong> 1200x675, JPEG, 85% quality</li>
                            <li><strong>Facebook:</strong> 1200x630, JPEG, 85% quality</li>
                            <li><strong>Web Gallery:</strong> 2048x2048, Auto format, 90% quality</li>
                            <li><strong>Email:</strong> 800x600, JPEG, 75% quality</li>
                        </ul>
                        
                        <h4>📝 Filename Patterns</h4>
                        <ul>
                            <li><code>{original}</code> - Original filename</li>
                            <li><code>{number}</code> - Sequential number (0001, 0002...)</li>
                            <li><code>{date}</code> - Current date (2024-03-15)</li>
                            <li><code>{dimensions}</code> - Image size (1920x1080)</li>
                            <li><code>{width}</code> / <code>{height}</code> - Individual dimensions</li>
                        </ul>
                        
                        <h4>⚠️ Tips & Best Practices</h4>
                        <ul>
                            <li>Always backup important images before processing</li>
                            <li>Use "Generate Report" for detailed analytics</li>
                            <li>Test settings on a small batch first</li>
                            <li>AI Protection is most effective with "Perlin" or "Frequency" methods</li>
                            <li>Watermark opacity of 0.3-0.4 provides good visibility without being intrusive</li>
                        </ul>
                    </div>
                    """)
            
            # Event handlers
            def validate_dirs(source, dest):
                """Validate directories and show status."""
                if not source and not dest:
                    return "<div class='status-box'>📋 Enter directories to validate</div>"
                
                is_valid, message = self.validate_inputs(source, dest)
                if is_valid:
                    return f"<div class='success-box'>{message}</div>"
                else:
                    return f"<div class='error-box'>{message}</div>"
            
            def scan_images_ui(source_dir):
                """Scan images and show preview."""
                if not source_dir:
                    return "📁 Enter a source directory to scan for images"
                
                image_files, stats = self.scan_images(source_dir)
                
                if not image_files:
                    error_msg = stats.get('error', 'No images found')
                    return f"❌ {error_msg}"
                
                # Format results
                formats_str = ", ".join([f"{ext}: {count}" for ext, count in stats['formats'].items()])
                estimated_time = self.estimate_processing_time(
                    stats['total_files'], 
                    stats['avg_size_mb'], 
                    ['resize', 'metadata']  # Default features
                )
                
                return f"""
                ✅ **Found {stats['total_files']} images**
                
                📊 **Details:**
                • Total size: {stats['total_size_mb']:.1f} MB
                • Average size: {stats['avg_size_mb']:.1f} MB per file
                • Formats: {formats_str}
                • Estimated processing time: {estimated_time}
                """
            
            def clear_all():
                """Reset all inputs to defaults."""
                return (
                    "", "", True, True, False, True, "Auto", 90, 1920, 1080,
                    False, False, True, "Full HD (1920x1080)", True, False,
                    "", False, "complete", 0.05, "perlin", "medium", False,
                    False, "", "bottom-right", 0.3, False, "subtle", "{original}",
                    "general", True, 
                    "<div class='status-box'>📋 Enter directories to validate</div>",
                    "📁 Enter a source directory to scan for images",
                    "🎯 **Ready to process images**\n\nChoose your source and destination directories, configure processing options, and click 'Process Images' to start."
                )
            
            def update_dimensions(preset):
                """Update dimensions when preset changes."""
                if preset != "Custom":
                    width, height = DIMENSION_PRESETS[preset]
                    return width, height
                return gr.update(), gr.update()
            
            # Wire up event handlers
            source_dir.change(
                fn=validate_dirs,
                inputs=[source_dir, dest_dir],
                outputs=dir_status
            )
            
            dest_dir.change(
                fn=validate_dirs,
                inputs=[source_dir, dest_dir],
                outputs=dir_status
            )
            
            scan_btn.click(
                fn=scan_images_ui,
                inputs=source_dir,
                outputs=scan_results
            )
            
            dimension_preset.change(
                fn=update_dimensions,
                inputs=dimension_preset,
                outputs=[max_width, max_height]
            )
            
            process_btn.click(
                fn=self.process_batch,
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
                    dir_status, scan_results, result_output
                ]
            )
            
            return app

def launch_app():
    """Launch the professional image batch processor."""
    try:
        processor = ImageBatchProcessor()
        app = processor.create_ui()
        
        # Launch with custom settings to avoid port conflicts
        app.launch(
            server_name="0.0.0.0",
            server_port=7861,  # Use different port to avoid conflicts
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            favicon_path=None
        )
        
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_app() 