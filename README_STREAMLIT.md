# 🎬 ClipQuery - Vivi RAG Chatbot (Streamlit Version)

A modern, web-based video analysis and clipping chatbot built with Streamlit, featuring RAG (Retrieval-Augmented Generation) for intelligent video content search and semantic video clipping.

## ✨ Features

### 🎯 Core Functionality
- **📹 Video Upload & Transcription**: Upload videos and automatically transcribe them using Groq Whisper API
- **🔍 Semantic Search**: RAG-powered search through video content using ChromaDB and sentence transformers
- **✂️ Intelligent Video Clipping**: Create video clips based on natural language queries
- **💬 AI Chat Interface**: Interactive chat with Vivi about video content
- **☁️ Google Drive Integration**: Sync videos with Google Drive (optional)

### 🎨 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Chat**: Live chat interface with message history
- **Video Player**: Built-in video playback for input and generated clips
- **Progress Indicators**: Visual feedback for transcription and processing
- **Debug Information**: Detailed insights into RAG operations

### 🧠 Advanced AI Features
- **Multi-Video Support**: Search across multiple video files
- **Context-Aware Clipping**: Intelligent clip boundaries that preserve complete thoughts
- **Acronym Detection**: Special handling for acronym explanations (NOPP, StarULIP, etc.)
- **Segment Merging**: Automatic merging of adjacent relevant segments
- **Natural Language Processing**: Understands complex queries and intent

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg installed and in PATH
- Groq API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ClipQuery
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Groq API key**:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   Or edit the key directly in `streamlit_vivi_rag.py` (line 107).

4. **Run the Streamlit app**:
   ```bash
   python run_streamlit_vivi.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_vivi_rag.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 🎬 Getting Started

1. **Upload a Video**: Use the sidebar to upload an MP4, MOV, or AVI file
2. **Wait for Transcription**: The app will automatically transcribe your video using Whisper
3. **Start Chatting**: Ask questions about the video content in the chat interface

### 💬 Chat Commands

#### General Questions
- `"What is NOPP?"` - Ask about specific concepts
- `"What are the main points discussed?"` - Get video summaries
- `"Explain the insurance coverage details"` - Request explanations

#### Video Clipping
- `"Clip: Show me the explanation of StarULIP"` - Create a video clip
- `"Clip: Insurance coverage details"` - Clip specific topics
- `"Clip: What is NOPP and how does it work?"` - Multi-part explanations

### 🎮 Interface Controls

#### Sidebar Features
- **📂 Upload Video**: Drag and drop or browse for video files
- **▶️ Play Input Video**: Watch the original uploaded video
- **🎬 Play Final Video**: View generated video clips
- **☁️ Google Drive Sync**: Manual sync with Google Drive
- **🔧 Debug Information**: View technical details

#### Chat Interface
- **Real-time Chat**: Type messages and get instant responses
- **Message History**: View previous conversations
- **Video Display**: Generated clips appear inline in chat
- **Progress Indicators**: See processing status

## 🔧 Technical Details

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Vivi RAG      │    │   Video RAG     │
│                 │◄──►│   Chatbot       │◄──►│   Pipeline      │
│ - Chat Interface│    │                 │    │                 │
│ - Video Upload  │    │ - LLM Processing│    │ - ChromaDB      │
│ - Video Player  │    │ - Clip Generation│   │ - Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FFmpeg        │    │   Groq Whisper  │    │   Google Drive  │
│   Processing    │    │   API           │    │   Sync          │
│                 │    │                 │    │                 │
│ - Video Clipping│    │ - Transcription │    │ - File Sync     │
│ - Concatenation │    │ - Timestamps    │    │ - Auto Upload   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

#### 1. StreamlitViviRAG Class
- **Core chatbot logic** from the original Tkinter version
- **RAG integration** for semantic search
- **Video processing** with FFmpeg
- **LLM interaction** with Groq API

#### 2. RAG Pipeline
- **ChromaDB vector store** for efficient similarity search
- **Sentence transformers** for text embeddings
- **Multi-video support** with video-specific collections
- **Acronym detection** and specialized handling

#### 3. Video Processing
- **Whisper transcription** with timestamps
- **Intelligent clipping** with natural boundaries
- **Video concatenation** for multi-clip outputs
- **Format compatibility** handling

### File Structure

```
ClipQuery/
├── streamlit_vivi_rag.py      # Main Streamlit application
├── run_streamlit_vivi.py      # Launcher script
├── rag_pipeline.py            # RAG implementation
├── google_drive_sync.py       # Google Drive integration
├── requirements.txt           # Python dependencies
├── README_STREAMLIT.md        # This file
├── Max Life Videos/           # Video storage directory
│   ├── video1.mp4
│   ├── video1.txt            # Transcripts
│   └── video1.srt            # Subtitle files
└── vector_store/             # ChromaDB storage
```

## 🎯 Advanced Features

### Intelligent Video Clipping

The system uses advanced algorithms to create meaningful video clips:

1. **Semantic Search**: Find relevant video segments using RAG
2. **Context Expansion**: Extend clips to include complete thoughts
3. **Natural Boundaries**: Start/end at sentence boundaries
4. **Segment Merging**: Combine adjacent relevant segments
5. **Acronym Handling**: Special logic for acronym explanations

### RAG Implementation

- **ChromaDB**: Fast vector similarity search
- **all-MiniLM-L6-v2**: Efficient sentence embeddings
- **Multi-collection**: Separate collections for videos and segments
- **Metadata enrichment**: Timestamps, video IDs, and similarity scores

### Multi-Video Support

- **Cross-video search**: Find content across multiple videos
- **Video-specific context**: Maintain video identity in responses
- **Aggregated results**: Combine relevant segments from multiple sources

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```bash
   # Install FFmpeg
   # Windows: Download from https://ffmpeg.org/download.html
   # macOS: brew install ffmpeg
   # Ubuntu: sudo apt install ffmpeg
   ```

2. **Groq API key issues**:
   - Ensure your API key is valid and has sufficient credits
   - Check the key format in the code

3. **Memory issues with large videos**:
   - Use smaller video files for testing
   - Ensure sufficient RAM for processing

4. **ChromaDB initialization**:
   - Delete `vector_store/` directory to reset embeddings
   - Re-run the app to rebuild the vector database

### Debug Information

The sidebar includes a debug section showing:
- Context length
- Number of transcript segments
- Current video path
- RAG operation details

## 🚀 Performance Optimization

### Tips for Better Performance

1. **Video Size**: Use compressed videos (H.264, reasonable bitrate)
2. **Batch Processing**: Process multiple videos during off-peak hours
3. **Memory Management**: Close unused browser tabs
4. **Network**: Ensure stable internet for API calls

### Scaling Considerations

- **Multiple Users**: Consider running multiple Streamlit instances
- **Large Video Libraries**: Implement video indexing and caching
- **API Limits**: Monitor Groq API usage and implement rate limiting

## 🔒 Security and Privacy

### Data Handling

- **Local Processing**: Videos are processed locally
- **Temporary Files**: Clip files are automatically cleaned up
- **API Security**: API keys should be stored securely
- **Google Drive**: Uses OAuth2 for secure authentication

### Best Practices

- Keep API keys secure and rotate regularly
- Monitor API usage and costs
- Implement user authentication for production use
- Regular backup of vector database

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install black flake8 pytest
   ```
4. **Make your changes**
5. **Run tests and linting**:
   ```bash
   black streamlit_vivi_rag.py
   flake8 streamlit_vivi_rag.py
   ```
6. **Submit a pull request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and modular

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Groq** for fast LLM inference
- **ChromaDB** for vector storage
- **FFmpeg** for video processing
- **Whisper** for speech recognition

## 📞 Support

For issues and questions:
1. Check the debug information in the app
2. Review the troubleshooting section
3. Open an issue on GitHub
4. Check the original Tkinter version for reference

---

**🎬 Happy video analyzing with Vivi!** 