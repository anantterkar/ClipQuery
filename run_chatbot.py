#!/usr/bin/env python3
"""
Helper script to run the ClipQuery chatbot with RAG pipeline
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'torch',
        'numpy',
        'langchain_ollama',
        'langchain_core',
        'customtkinter',
        'openai-whisper',
        'moviepy',
        'pytubefix',
        'opencv-python',
        'pillow',
        'httpx'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print(f"   {sys.executable} -m pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed!")
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            return True
    except:
        pass
    
    print("⚠️  Ollama is not running or not accessible.")
    print("   Please start Ollama and ensure the 'gemma3' model is available.")
    print("   You can install Ollama from: https://ollama.ai/")
    print("   Then run: ollama pull gemma3")
    return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is available!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg is not found in PATH.")
        print("   Please install FFmpeg from: https://ffmpeg.org/download.html")
        print("   Or use a package manager like:")
        print("   - Windows: choco install ffmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")
        return False

def check_transcripts():
    """Check if transcript files exist"""
    transcripts_folder = "Max Life Videos"
    if os.path.exists(transcripts_folder):
        txt_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.txt')]
        if txt_files:
            print(f"✅ Found {len(txt_files)} transcript files in '{transcripts_folder}' folder")
            return True
        else:
            print(f"⚠️  No .txt transcript files found in '{transcripts_folder}' folder")
    else:
        print(f"⚠️  Transcripts folder '{transcripts_folder}' not found")
    
    print("   The RAG pipeline will initialize with empty data.")
    print("   You can add transcript files later.")
    return False

def main():
    print("🎬 ClipQuery Chatbot Setup Check")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Check Ollama
    if not check_ollama():
        print("   You can still run the chatbot, but LLM features won't work.")
    
    print()
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("   You can still run the chatbot, but video processing won't work.")
    
    print()
    
    # Check transcripts
    check_transcripts()
    
    print()
    print("🚀 Starting ClipQuery Chatbot...")
    print("=" * 40)
    
    try:
        from Chatbot2 import ViviChatbot
        app = ViviChatbot()
        app.run()
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Please check the error message above and fix any issues.")

if __name__ == "__main__":
    main() 