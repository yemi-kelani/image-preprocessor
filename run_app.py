#!/usr/bin/env python3
"""
Launcher script for the Professional Image Batch Processor

This script provides a convenient way to launch the application with proper
error handling and configuration.
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main launcher function."""
    print("🎨 Professional Image Batch Processor")
    print("=" * 50)
    
    try:
        # Import and launch the application
        from app import launch_app
        
        print("🚀 Starting application...")
        print("📱 The app will open in your default browser")
        print("🌐 If it doesn't open automatically, go to: http://localhost:7861")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        launch_app()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        sys.exit(0)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 