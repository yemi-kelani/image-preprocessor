# Troubleshooting Guide

## Common Issues and Solutions

### 1. `ModuleNotFoundError: No module named 'gradio'`

**Problem**: Gradio is not installed or not available in the current Python environment.

**Solutions**:
```bash
# Option 1: Install dependencies manually
pip install -r requirements.txt

# Option 2: Use the launcher (recommended)
python run.py

# Option 3: Install specific versions
pip install gradio>=4.0.0 Pillow>=10.0.0 "numpy>=1.24.0,<2.0.0"
```

### 2. NumPy Version Conflicts

**Problem**: Error messages about numpy version conflicts with other packages.

**Solution**:
```bash
# Fix numpy version to be compatible
pip install --force-reinstall "numpy>=1.24.0,<2.0.0"
pip install gradio Pillow
```

### 3. Application Won't Start

**Problem**: The application fails to launch.

**Steps to diagnose**:
1. **Check dependencies**:
   ```bash
   python -c "import gradio, PIL, numpy; print('All dependencies OK')"
   ```

2. **Check import**:
   ```bash
   python -c "from image_processor import app; print('App imports OK')"
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.8 or higher
   ```

### 4. Environment Issues

**Problem**: Using conda/virtual environments.

**Solutions**:
```bash
# If using conda
conda activate your_environment
pip install -r requirements.txt

# If using virtual environment
source your_venv/bin/activate  # Linux/Mac
# or
your_venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 5. Permission Errors

**Problem**: Permission denied when accessing directories.

**Solutions**:
- Ensure you have read access to source directories
- Ensure you have write access to destination directories
- On macOS/Linux, use `chmod` to adjust permissions if needed

### 6. Memory Issues with Large Images

**Problem**: Application crashes or runs out of memory.

**Solutions**:
- Process smaller batches of images
- Reduce the maximum image dimensions
- Close other applications to free up RAM

## Performance Tips

1. **For large batches**: Process in smaller chunks
2. **For high-resolution images**: Use smaller max dimensions (e.g., 1920x1080)
3. **For faster processing**: Disable unnecessary features (metadata stripping, noise addition)

## Getting Help

If you encounter issues not covered here:
1. Check the error message in the terminal
2. Verify all dependencies are installed correctly
3. Try the manual installation steps
4. Ensure you have sufficient disk space and memory 