#!/usr/bin/env python
"""
Setup Helper Script for Enterprise PDF Knowledge Application
Helps with installation, testing, and initial configuration
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_step(step_num: int, text: str):
    """Print step indicator."""
    print(f"[{step_num}/5] {text}...")

def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")

def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required, but {version.major}.{version.minor} found")
        return False

def create_virtual_env() -> bool:
    """Create virtual environment."""
    print_step(1, "Creating virtual environment")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            check=True,
            capture_output=True
        )
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python() -> str:
    """Get path to virtual environment Python."""
    if sys.platform == "win32":
        return "venv\\Scripts\\python"
    return "venv/bin/python"

def install_requirements() -> bool:
    """Install Python dependencies."""
    print_step(2, "Installing Python dependencies")
    venv_python = get_venv_python()
    
    try:
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True
        )
        print_success("pip upgraded")
        
        subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            timeout=300
        )
        print_success("All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False
    except subprocess.TimeoutExpired:
        print_error("Installation timed out (> 5 minutes)")
        return False

def check_tesseract() -> bool:
    """Check if Tesseract is installed."""
    print_step(3, "Checking Tesseract OCR")
    
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print_success(f"Tesseract found: {version_line}")
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        pass
    
    print_warning("Tesseract OCR not found")
    print("  Installation instructions:")
    if sys.platform == "linux":
        print("    Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    elif sys.platform == "darwin":
        print("    macOS: brew install tesseract")
    elif sys.platform == "win32":
        print("    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    return False

def setup_env_file() -> bool:
    """Set up .env file."""
    print_step(4, "Setting up configuration")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print_warning(".env already exists, skipping")
        return True
    
    print("\nTo get your Groq API key:")
    print("  1. Visit https://console.groq.com")
    print("  2. Sign up or login")
    print("  3. Go to API Keys section")
    print("  4. Create a new API key")
    
    api_key = input("\nEnter your Groq API key (or press Enter to skip): ").strip()
    
    if api_key:
        env_file.write_text(f"GROQ_API_KEY={api_key}\n")
        print_success(".env file created with API key")
        return True
    else:
        print_warning("Skipped .env setup (you can add it manually later)")
        return True

def create_data_directories() -> bool:
    """Create data directory structure."""
    print_step(5, "Creating data directories")
    
    dirs = ["data", "data/docs", "data/tables", "data/images"]
    
    try:
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print_success("Data directories created")
        return True
    except Exception as e:
        print_error(f"Failed to create directories: {e}")
        return False

def test_streamlit() -> bool:
    """Test Streamlit installation."""
    venv_python = get_venv_python()
    
    try:
        result = subprocess.run(
            [venv_python, "-c", "import streamlit; print(streamlit.__version__)"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print_success(f"Streamlit {result.stdout.strip()} installed")
            return True
    except Exception as e:
        print_error(f"Streamlit test failed: {e}")
    
    return False

def run_setup_wizard():
    """Main setup wizard."""
    print_header("📚 Enterprise PDF → Knowledge Application Setup")
    
    print("This wizard will help you set up the application.\n")
    
    # Check prerequisites
    if not check_python_version():
        print_error("Setup failed: Python 3.8+ is required")
        return False
    
    # Create virtual environment
    if not create_virtual_env():
        return False
    
    # Install dependencies
    if not install_requirements():
        print_warning("Dependency installation had issues, but continuing...")
    
    # Test Streamlit
    if not test_streamlit():
        print_error("Streamlit installation verification failed")
        return False
    
    # Check Tesseract
    has_tesseract = check_tesseract()
    
    # Setup environment
    if not setup_env_file():
        return False
    
    # Create directories
    if not create_data_directories():
        return False
    
    # Print final summary
    print_header("✅ Setup Complete!")
    
    if sys.platform == "win32":
        activation_cmd = "venv\\Scripts\\activate"
    else:
        activation_cmd = "source venv/bin/activate"
    
    print("Next steps:\n")
    print(f"1. Activate virtual environment:")
    print(f"   $ {activation_cmd}\n")
    print(f"2. Run the application:")
    print(f"   $ streamlit run app_enhanced.py\n")
    print(f"3. Open browser to: http://localhost:8501\n")
    
    if not has_tesseract:
        print("⚠️  Optional: Install Tesseract for better OCR support")
        if sys.platform == "linux":
            print("   $ sudo apt-get install tesseract-ocr")
        elif sys.platform == "darwin":
            print("   $ brew install tesseract")
        print()
    
    print("For more information, see README.md\n")
    
    return True

def main():
    """Main entry point."""
    try:
        success = run_setup_wizard()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
