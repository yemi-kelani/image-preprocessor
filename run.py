#!/usr/bin/env python3
"""
Simple launcher for the Image Batch Processor
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def main():
    print("🖼️  Image Batch Processor Launcher")
    print("=" * 40)
    
    # Check if requirements are installed
    try:
        import gradio
        import PIL
        import numpy
        print("✅ All dependencies are already installed!")
    except ImportError:
        print("📦 Installing required dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return
    
    # Launch the application
    print("\n🚀 Launching Image Batch Processor...")
    print("🌐 The app will be available at: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Import and run the app
        from image_processor import app
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("🔧 Try running 'python image_processor.py' directly")

if __name__ == "__main__":
    main() 