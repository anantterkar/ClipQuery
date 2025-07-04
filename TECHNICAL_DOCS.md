# ClipQuery Technical Documentation

<div align="center">
  <h1>🔧 Technical Documentation</h1>
  <p><strong>Comprehensive Guide to ClipQuery's Architecture & Implementation</strong></p>
  <p><em>Updated for Vivi_RAG_final3.py - December 2024</em></p>
</div>

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Main Application Class](#main-application-class)
4. [RAG Pipeline](#rag-pipeline)
5. [Google Drive Integration](#google-drive-integration)
6. [Video Processing](#video-processing)
7. [AI/LLM Integration](#aimllm-integration)
8. [GUI Components](#gui-components)
9. [Utility Functions](#utility-functions)
10. [Performance Optimization](#performance-optimization)
11. [Error Handling](#error-handling)
12. [Configuration](#configuration)
13. [Function Reference](#function-reference)
14. [System Workflows](#system-workflows)
15. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Overview

ClipQuery is built on a modular architecture that combines multiple technologies to create an intelligent video analysis system. The core components work together to provide seamless video search, analysis, and clip generation capabilities.

### Technology Stack
- **Frontend**: CustomTkinter (Modern GUI Framework)
- **AI Engine**: Groq LLM (Llama-3.3-70B)
- **Search Engine**: RAG Pipeline with FAISS
- **Storage**: Google Drive API + Local Cache
- **Video Processing**: FFmpeg
- **Transcription**: OpenAI Whisper
- **Vector Database**: FAISS (Facebook AI Similarity Search)

---

## 🧠 Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ViviChatbot Class                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   GUI Layer │  │  AI Engine  │  │ Video Proc  │        │
│  │             │  │             │  │             │        │
│  │ • Chat UI   │  │ • LLM Calls │  │ • FFmpeg    │        │
│  │ • Buttons   │  │ • RAG Query │  │ • Clipping  │        │
│  │ • Progress  │  │ • Context   │  │ • Concaten. │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Google Drive │  │  RAG Pipe   │  │  Utilities  │        │
│  │   Sync      │  │             │  │             │        │
│  │             │  │ • Vector DB │  │ • Time Parse│        │
│  │ • Auth      │  │ • Search    │  │ • File Ops  │        │
│  │ • Download  │  │ • Similarity│  │ • Validation│        │
│  │ • Upload    │  │ • Filtering │  │ • Debug     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Main Application Class

### ViviChatbot Class

The main application class that orchestrates all system components.

#### Constructor (`__init__`)
```python
    def __init__(self):
        # Initialize GUI components
        # Set up Google Drive sync
        # Initialize RAG pipeline
    # Configure conversation management
```

**Key Initialization Steps:**
1. **GUI Setup**: Configure CustomTkinter with light theme
2. **Google Drive**: Initialize background sync thread
3. **RAG Pipeline**: Set up vector database and search
4. **Conversation Management**: Configure history limits
5. **Missing Video Check**: Scan for incomplete downloads

#### Core Properties
- `self.root`: Main GUI window
- `self.drive_sync`: Google Drive synchronization object
- `self.rag`: RAG pipeline for semantic search
- `self.conversation_history`: Chat history management
- `self.similarity_threshold`: Search relevance threshold (0.75)

---

## 🔍 RAG Pipeline Integration

### VideoRAG Class (External)
The RAG pipeline provides semantic search capabilities across video transcripts.

#### Key Methods:
```python
def query_videos(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search across video transcripts.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return
        
    Returns:
        List of dictionaries containing:
        - video_id: Video identifier
        - start: Start time (seconds)
        - end: End time (seconds)
        - text: Transcript segment
        - similarity: Relevance score
    """
```

#### Search Process:
1. **Query Preprocessing**: Normalize and enhance search terms
2. **Vector Search**: Use FAISS for similarity matching
3. **Result Filtering**: Apply relevance thresholds
4. **Context Building**: Aggregate relevant segments

---

## ☁️ Google Drive Integration

### GoogleDriveSync Class

Handles all Google Drive operations including authentication, file synchronization, and missing video detection.

#### Authentication Flow
```python
def authenticate(self):
    """
    Authenticate with Google Drive API using OAuth 2.0.
    Handles credential refresh and token management.
    """
```

#### File Synchronization
```python
def list_drive_files(self) -> List[Dict[str, Any]]:
    """
    List all files in Google Drive folder with pagination support.
    Handles folders with >100 files by implementing proper pagination.
    """
```

**Pagination Implementation:**
- Uses `pageToken` and `nextPageToken` for large folders
- Maximum page size of 1000 items per request
- Automatic retry logic for failed requests
- Comprehensive error handling

#### Missing Video Detection
```python
def check_for_missing_videos(self):
    """
    Identify videos that have transcripts but missing MP4 files.
    Triggers automatic download attempts.
    """
```

---

## 🎬 Video Processing

### FFmpeg Integration

The system uses FFmpeg for all video processing operations including clipping, concatenation, and format conversion.

#### Video Clipping
```python
def clip_video(self, video_path, start_time, end_time):
    """
    Create video clips using FFmpeg with optimized settings.
    
    Features:
    - H.264 video codec with CRF 23 quality
    - AAC audio codec at 44.1kHz
    - 30fps frame rate standardization
    - Fast encoding preset for speed
    - Optimized for streaming (faststart)
    """
```

**FFmpeg Command Structure:**
```bash
ffmpeg -y -ss {start_time} -i {video_path} -t {duration} \
       -c:v libx264 -c:a aac -preset fast -crf 23 \
       -r 30 -ar 44100 -ac 2 -movflags +faststart \
       -avoid_negative_ts make_zero -fflags +genpts \
       {output_path}
```

#### Video Concatenation
```python
def concatenate_videos(video_paths, output_filepath):
    """
    Concatenate multiple video clips into a single output file.
    Handles codec compatibility and maintains quality.
    """
```

**Concatenation Process:**
1. **Compatibility Check**: Validate video properties
2. **List File Creation**: Generate FFmpeg concat list
3. **Re-encoding**: Ensure consistent codec settings
4. **Cleanup**: Remove temporary files

---

## 🤖 AI/LLM Integration

### Groq LLM Integration

The system uses Groq's Llama-3.3-70B model for natural language processing and video analysis.

#### LLM Configuration
```python
# LLM instances for different tasks
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
llm_clipping = ChatGroq(model="llama-3.3-70b-versatile", temperature=1.2)
```

#### Prompt Templates

**General Conversation Template:**
```python
general_template = """
You are Vivi, an expert and friendly video assistant chatbot.
You have access to video transcripts and can answer questions about the content.
Handle acronyms and technical terms with variations in spacing and casing.
"""
```

**Clipping Template:**
```python
clipping_template = """
Create comprehensive video clips that fully answer user queries.
Focus on complete explanations, especially for acronyms and multi-part concepts.
Ensure clips start and end at natural points.
"""
```

#### Context Management
```python
def build_llm_context(self, query, for_clipping=False):
    """
    Build context for LLM by combining RAG results and conversation history.
    
    Features:
    - Multi-query detection and handling
    - Context truncation for token limits
    - Relevance-based video selection
    - Conversation history integration
    """
```

---

## 🖥️ GUI Components

### CustomTkinter Interface

The GUI is built using CustomTkinter for a modern, responsive interface.

#### Main Window Structure
```python
# Window configuration
self.root = ctk.CTk()
self.root.configure(fg_color="#FFFFFF")  # Light theme
self.root.title("ClipQuery- Video Chatbot")
self.root.geometry("900x650")
```

#### Component Hierarchy
```
Root Window
├── Header Frame (Logo + Title)
├── Chat Frame (Scrollable)
├── Entry Frame (Input + Send Button)
├── Button Frame (Video + Sync)
└── Progress Bar (Hidden by default)
```

#### Chat Display System
```python
def display_message(self, sender, message):
    """
    Display messages in chat interface with proper styling.
    
    Features:
    - Different colors for user vs assistant
    - Proper text wrapping
    - Responsive layout
    - Auto-scroll to bottom
    """
```

#### Typewriter Effect
```python
def typewriter_effect(self, sender, message):
    """
    Animate text appearance for more engaging user experience.
    Displays text word by word with timing control.
    """
```

---

## 🔧 Utility Functions

### Time Management
```python
def _parse_srt_time(t):
    """
    Parse SRT timestamp formats into seconds.
    Handles multiple formats: HH:MM:SS,mmm, MM:SS, and float seconds.
    """

def format_time(seconds):
    """
    Convert seconds to SRT format: HH:MM:SS,mmm
    Used for subtitle generation and time display.
    """
```

### Token Management
```python
def estimate_tokens(self, text):
    """
    Rough token estimation (1 token ≈ 4 characters).
    Used for context length management.
    """

def check_token_limit(self, text, max_tokens=8000):
    """
    Ensure text doesn't exceed LLM token limits.
    Truncates with ellipsis if necessary.
    """
```

### Conversation History
```python
def manage_conversation_history(self, user_input, response):
    """
    Manage conversation history to prevent token overflow.
    
    Features:
    - Automatic truncation
    - Turn-based management
    - Context rebuilding
    - Memory optimization
    """
```

---

## ⚡ Performance Optimization

### Memory Management
- **Conversation History**: Limited to 10 turns maximum
- **Context Truncation**: 8000 character limit for context
- **Auto-clear**: Automatic history clearing when memory usage is high
- **Token Estimation**: Proactive token limit checking

### Video Processing Optimization
- **Parallel Processing**: Background thread execution
- **Temporary File Management**: Automatic cleanup of clip files
- **Codec Optimization**: Fast encoding presets
- **Quality Settings**: Balanced quality vs. speed (CRF 23)

### Search Optimization
- **Similarity Threshold**: 0.75 for relevance filtering
- **Result Limiting**: Maximum 8 results for normal queries
- **Pagination**: Efficient Google Drive API usage
- **Caching**: Local file caching to reduce API calls

---

## 🛡️ Error Handling

### Comprehensive Error Management
```python
def ensure_video_file_available(self, video_id):
    """
    Robust video file availability checking with fallback mechanisms.
    
    Error Handling:
    - File not found → Google Drive download attempt
    - Download failure → Alternative video search
    - No alternatives → Fallback to available videos
    - Complete failure → User notification
    """
```

### Google Drive Error Recovery
- **Authentication Failures**: Automatic token refresh
- **Network Issues**: Retry logic with exponential backoff
- **Permission Errors**: Clear error messages and guidance
- **API Limits**: Rate limiting and quota management

### Video Processing Error Recovery
- **FFmpeg Failures**: Detailed error logging and user feedback
- **File Corruption**: Integrity checking and re-download
- **Codec Issues**: Automatic format conversion
- **Memory Issues**: Graceful degradation and cleanup

---

## ⚙️ Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY=gsk_your_api_key_here

# Optional
GOOGLE_DRIVE_CREDENTIALS=credentials.json
```

### Application Settings
```python
# Conversation Management
self.max_conversation_turns = 10
self.max_context_chars = 8000

# Search Configuration
self.similarity_threshold = 0.75

# Video Processing
MAX_CLIP_DURATION = 60.0  # seconds
VIDEO_QUALITY_CRF = 23
FRAME_RATE = 30
AUDIO_SAMPLE_RATE = 44100
```

### Performance Tuning
```python
# RAG Configuration
RAG_RESULTS_NORMAL = 8
RAG_RESULTS_CLIPPING = 15

# Google Drive
PAGE_SIZE = 1000
SYNC_INTERVAL = 30  # seconds

# Memory Management
TOKEN_LIMIT_NORMAL = 8000
TOKEN_LIMIT_CLIPPING = 8000
```

---

## 📊 Monitoring & Debugging

### Debug Functions
```python
def debug_video_usage(self, rag_results, query_type="normal"):
    """
    Comprehensive debugging for video search and usage.
    Shows detailed information about RAG results and video assignments.
    """

def debug_google_drive(self):
    """
    Debug Google Drive setup and permissions.
    Lists folders, files, and identifies missing videos.
    """
```

### Logging System
- **Console Output**: Detailed progress and error messages
- **Status Indicators**: Real-time sync and processing status
- **Error Tracking**: Comprehensive error logging with context
- **Performance Metrics**: Timing and resource usage tracking

---

## 🔄 System Workflows

### Normal Query Workflow
1. **User Input**: Natural language question
2. **RAG Search**: Semantic search across video transcripts
3. **Context Building**: Aggregate relevant video segments
4. **LLM Processing**: Generate comprehensive response
5. **Response Display**: Show answer with conversation history update

### Clipping Query Workflow
1. **Query Analysis**: Parse clipping request
2. **Multi-Video Search**: Find relevant segments across videos
3. **LLM Clip Generation**: Generate precise time ranges
4. **Video Assignment**: Match clips to correct videos
5. **Clip Creation**: Use FFmpeg to create video clips
6. **Concatenation**: Combine multiple clips if needed
7. **Output**: Save final video to Downloads folder

### Google Drive Sync Workflow
1. **Authentication**: OAuth 2.0 authentication
2. **File Discovery**: List all files with pagination
3. **Missing Detection**: Identify videos without MP4 files
4. **Download Queue**: Prioritize missing video downloads
5. **Background Sync**: Continuous monitoring for changes
6. **Error Recovery**: Handle network and permission issues

---

## 🚀 Future Enhancements

### Planned Features
- **Web Interface**: Streamlit-based web UI
- **Batch Processing**: Bulk video analysis
- **Advanced Analytics**: Usage statistics and insights
- **Plugin System**: Extensible architecture for custom features
- **Cloud Deployment**: Docker containerization
- **API Endpoints**: RESTful API for integration

### Performance Improvements
- **GPU Acceleration**: CUDA support for video processing
- **Distributed Processing**: Multi-node video analysis
- **Advanced Caching**: Redis-based result caching
- **Streaming**: Real-time video processing
- **Compression**: Advanced video compression algorithms

---

<div align="center">
  <p><strong>Technical Documentation v2.0</strong></p>
  <p>Last updated: December 2024</p>
</div>

<style>
/* Modern CSS styling for technical documentation */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

h1, h2, h3, h4, h5, h6 {
    color: #0366d6;
    font-weight: 600;
    margin-top: 24px;
    margin-bottom: 16px;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
}

h1 {
    font-size: 2.5em;
    text-align: center;
    background: linear-gradient(45deg, #0366d6, #28a745);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    border-bottom: none;
}

h2 {
    font-size: 1.8em;
    color: #24292e;
    background: #f6f8fa;
    padding: 10px 15px;
    border-radius: 6px;
    border-left: 4px solid #0366d6;
}

h3 {
    font-size: 1.4em;
    color: #24292e;
    border-bottom: 2px solid #e1e4e8;
}

code {
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    font-size: 85%;
    margin: 0;
    padding: 0.2em 0.4em;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

pre {
    background-color: #f6f8fa;
    border-radius: 6px;
    font-size: 85%;
    line-height: 1.45;
    overflow: auto;
    padding: 16px;
    border: 1px solid #e1e4e8;
}

pre code {
    background-color: transparent;
    border: 0;
    display: inline;
    line-height: inherit;
    margin: 0;
    overflow: visible;
    padding: 0;
    word-wrap: normal;
}

blockquote {
    border-left: 0.25em solid #dfe2e5;
    color: #6a737d;
    margin: 0;
    padding: 0 1em;
    background: #f6f8fa;
    border-radius: 0 6px 6px 0;
}

table {
    border-collapse: collapse;
    border-spacing: 0;
    width: 100%;
    overflow: auto;
    display: block;
    margin: 20px 0;
}

table th, table td {
    border: 1px solid #dfe2e5;
    padding: 8px 12px;
    text-align: left;
}

table th {
    background-color: #f6f8fa;
    font-weight: 600;
    color: #24292e;
}

table tr:nth-child(even) {
    background-color: #f8f9fa;
}

table tr:hover {
    background-color: #f1f3f4;
}

img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.badge {
    display: inline-block;
    padding: 0.25em 0.4em;
    font-size: 75%;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.25rem;
    margin: 0 2px;
}

.badge-blue { background-color: #007bff; color: white; }
.badge-green { background-color: #28a745; color: white; }
.badge-orange { background-color: #fd7e14; color: white; }
.badge-yellow { background-color: #ffc107; color: #212529; }
.badge-purple { background-color: #6f42c1; color: white; }

.architecture-diagram {
    background: white;
    border: 2px solid #e1e4e8;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.2;
}

.workflow-step {
    background: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 6px 6px 0;
}

.workflow-step h4 {
    margin-top: 0;
    color: #28a745;
    border-bottom: none;
}

.feature-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.feature-box h3 {
    color: white;
    border-bottom: 1px solid rgba(255,255,255,0.3);
}

.alert {
    padding: 12px 16px;
    border-radius: 6px;
    margin: 16px 0;
    border-left: 4px solid;
}

.alert-info {
    background-color: #d1ecf1;
    border-color: #17a2b8;
    color: #0c5460;
}

.alert-warning {
    background-color: #fff3cd;
    border-color: #ffc107;
    color: #856404;
}

.alert-success {
    background-color: #d4edda;
    border-color: #28a745;
    color: #155724;
}

.alert-danger {
    background-color: #f8d7da;
    border-color: #dc3545;
    color: #721c24;
}

---

## 📚 Function Reference

### Core Application Functions

#### `ViviChatbot.__init__()`
**Purpose**: Initialize the main application
**Key Operations**:
- Set up CustomTkinter GUI with light theme
- Initialize Google Drive sync in background thread
- Configure conversation history management
- Set up RAG pipeline for semantic search
- Check for missing video files on startup

#### `ViviChatbot.send_message()`
**Purpose**: Handle user input and generate responses
**Workflow**:
1. Parse user input for clipping vs. normal queries
2. Build LLM context using RAG pipeline
3. Generate AI response using Groq LLM
4. For clipping queries: create video clips using FFmpeg
5. Update conversation history and display results

#### `ViviChatbot.build_llm_context(query, for_clipping=False)`
**Purpose**: Build context for LLM by combining RAG results and conversation history
**Features**:
- Multi-query detection and handling
- Context truncation for token limits
- Relevance-based video selection
- Conversation history integration
- Enhanced query preprocessing for better RAG results

### Video Processing Functions

#### `ViviChatbot.clip_video(video_path, start_time, end_time)`
**Purpose**: Create video clips using FFmpeg
**Optimizations**:
- H.264 video codec with CRF 23 quality
- AAC audio codec at 44.1kHz
- 30fps frame rate standardization
- Fast encoding preset for speed
- Optimized for streaming (faststart)

#### `concatenate_videos(video_paths, output_filepath)`
**Purpose**: Concatenate multiple video clips into a single output file
**Process**:
1. Validate video compatibility
2. Create FFmpeg concat list file
3. Re-encode for consistent codec settings
4. Clean up temporary files

#### `ViviChatbot.get_video_duration(video_path)`
**Purpose**: Get video duration using OpenCV
**Returns**: Duration in seconds or None if error

#### `ViviChatbot.get_video_properties(video_path)`
**Purpose**: Get comprehensive video properties using FFprobe
**Returns**: Dictionary with codec, resolution, frame rate, etc.

### RAG and Search Functions

#### `ViviChatbot.load_transcript_segments(video_id, relevant_ranges=None)`
**Purpose**: Load transcript segments for a video
**Features**:
- Optional filtering by time ranges
- Robust error handling
- Support for multiple timestamp formats

#### `ViviChatbot.load_full_transcript(video_id)`
**Purpose**: Load complete transcript for a video
**Returns**: Full transcript text or empty string if error

#### `ViviChatbot.improve_rag_results(rag_results, query)`
**Purpose**: Enhance RAG results with exact keyword matching
**Improvements**:
- Exact word match boosting
- Phrase match detection
- Technical term recognition
- Video content relevance scoring

### Google Drive Integration Functions

#### `ViviChatbot.init_google_drive_sync()`
**Purpose**: Initialize Google Drive sync in background thread
**Features**:
- Background thread execution
- Automatic initial sync
- Error handling and status updates
- File watching for changes

#### `ViviChatbot.ensure_video_file_available(video_id)`
**Purpose**: Ensure video file is available locally
**Fallback Strategy**:
1. Check local file existence
2. Attempt Google Drive download
3. Find alternative videos
4. Provide detailed error feedback

#### `ViviChatbot.find_alternative_video_for_clip(clip, rag_results)`
**Purpose**: Find alternative video when original is missing
**Process**:
1. Search for videos with similar content
2. Check for available MP4 files
3. Update clip metadata
4. Return best alternative

### Conversation Management Functions

#### `ViviChatbot.manage_conversation_history(user_input, response)`
**Purpose**: Manage conversation history to prevent token overflow
**Features**:
- Automatic truncation at 10 turns
- Context character limit of 8000
- Token estimation and checking
- Auto-clear when memory usage is high

#### `ViviChatbot.estimate_tokens(text)`
**Purpose**: Rough token estimation (1 token ≈ 4 characters)
**Usage**: Context length management and LLM preparation

#### `ViviChatbot.check_token_limit(text, max_tokens=8000)`
**Purpose**: Ensure text doesn't exceed LLM token limits
**Action**: Truncates with ellipsis if necessary

### GUI and Display Functions

#### `ViviChatbot.display_message(sender, message)`
**Purpose**: Display messages in chat interface
**Styling**:
- Different colors for user vs assistant
- Proper text wrapping
- Responsive layout
- Auto-scroll to bottom

#### `ViviChatbot.typewriter_effect(sender, message)`
**Purpose**: Animate text appearance for engaging UX
**Features**:
- Word-by-word display
- Configurable timing
- Smooth animation
- Auto-scroll during animation

#### `ViviChatbot.show_buffering()` / `ViviChatbot.hide_buffering()`
**Purpose**: Show/hide processing indicators
**Animation**: Animated dots during processing

### Utility Functions

#### `_parse_srt_time(t)`
**Purpose**: Parse SRT timestamp formats into seconds
**Supported Formats**:
- HH:MM:SS,mmm
- MM:SS
- Float seconds

#### `format_time(seconds)`
**Purpose**: Convert seconds to SRT format: HH:MM:SS,mmm
**Usage**: Subtitle generation and time display

#### `ViviChatbot.normalize(text)`
**Purpose**: Normalize text for comparison (remove spaces, lowercase)
**Usage**: Acronym and term matching

### Clip Processing Functions

#### `ViviChatbot.parse_llm_clip_response(response)`
**Purpose**: Parse LLM response to extract clip ranges
**Features**:
- Multiple pattern matching
- Video ID assignment
- Time range validation
- Duration limits (max 300 seconds)

#### `ViviChatbot.assign_clips_to_correct_videos(clips, query, rag_results)`
**Purpose**: Assign clips to correct videos based on content relevance
**Strategy**:
- Respect LLM video assignments
- Content-based matching
- Overlap calculation
- Fallback mechanisms

#### `ViviChatbot.deduplicate_clips(clips, min_overlap=0.3, query="")`
**Purpose**: Remove overlapping clips and keep most relevant
**Features**:
- Multi-topic query handling
- Overlap ratio calculation
- Video-based deduplication
- Conservative approach for different videos

#### `ViviChatbot.validate_and_improve_clips(clips, video_duration, is_acronym_query=False, query="")`
**Purpose**: Validate clip ranges and ensure they're within video bounds
**Validation**:
- Start < end time
- Within video duration
- No negative values

### Debug and Monitoring Functions

#### `ViviChatbot.debug_video_usage(rag_results, query_type="normal")`
**Purpose**: Comprehensive debugging for video search and usage
**Output**:
- Video distribution analysis
- Segment relevance scores
- Time range information
- Content preview

#### `ViviChatbot.check_for_missing_videos()`
**Purpose**: Check for videos with transcripts but no MP4 files
**Returns**: List of missing video IDs

#### `ViviChatbot.check_missing_videos_on_startup()`
**Purpose**: Check for missing videos on startup and notify user
**Features**:
- Background thread execution
- User-friendly notifications
- Actionable guidance

---

## 🔄 System Workflows

### Normal Query Workflow
1. **User Input**: Natural language question
2. **Query Analysis**: Determine query type and intent
3. **RAG Search**: Semantic search across video transcripts
4. **Context Building**: Aggregate relevant video segments
5. **LLM Processing**: Generate comprehensive response using Groq
6. **Response Display**: Show answer with conversation history update

### Clipping Query Workflow
1. **Query Analysis**: Parse clipping request and extract terms
2. **Multi-Video Search**: Find relevant segments across all videos
3. **LLM Clip Generation**: Generate precise time ranges using AI
4. **Video Assignment**: Match clips to correct videos based on content
5. **Deduplication**: Remove overlapping clips
6. **Clip Creation**: Use FFmpeg to create individual video clips
7. **Concatenation**: Combine multiple clips if needed
8. **Output**: Save final video to Downloads folder

### Google Drive Sync Workflow
1. **Authentication**: OAuth 2.0 authentication with token refresh
2. **File Discovery**: List all files with pagination support
3. **Missing Detection**: Identify videos without MP4 files
4. **Download Queue**: Prioritize missing video downloads
5. **Background Sync**: Continuous monitoring for changes
6. **Error Recovery**: Handle network and permission issues

### Conversation Management Workflow
1. **Input Processing**: Add new turn to conversation history
2. **History Truncation**: Keep only last 10 turns
3. **Context Building**: Rebuild context from history
4. **Token Checking**: Ensure context fits within limits
5. **Auto-clear**: Clear history if memory usage is high

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Google Drive Sync Problems
**Symptoms**: Files not found, sync failures
**Solutions**:
```bash
# Check credentials
python debug_google_drive.py

# Reset authentication
rm token.json
python Vivi_RAG_final3.py

# Verify folder permissions
# Check Google Drive API quotas
```

#### Video Processing Errors
**Symptoms**: FFmpeg failures, corrupted clips
**Solutions**:
```bash
# Verify FFmpeg installation
ffmpeg -version

# Check video file integrity
python -c "import cv2; print('OpenCV available')"

# Clear temporary files
rm -rf /tmp/clip_*
```

#### Memory Issues
**Symptoms**: Slow performance, crashes
**Solutions**:
- Reduce `max_conversation_turns` in settings
- Close other applications
- Increase system RAM
- Clear conversation history manually

#### RAG Search Issues
**Symptoms**: Poor search results, missing content
**Solutions**:
- Check transcript file integrity
- Verify vector database
- Adjust similarity threshold
- Rebuild RAG pipeline

### Performance Optimization Tips

1. **Storage**: Use SSD for faster video processing
2. **Network**: Ensure stable internet for Google Drive sync
3. **Memory**: Close unnecessary applications
4. **Processing**: Use background threads for heavy operations
5. **Caching**: Leverage local file caching

### Debug Commands

```bash
# Terminal mode for testing
python Vivi_RAG_final3.py "your query here"

# Debug Google Drive setup
python debug_google_drive.py

# Check system requirements
python -c "import cv2, whisper, customtkinter; print('All dependencies OK')"
```

---

<div align="center">
  <p><strong>Technical Documentation v2.0</strong></p>
  <p>Last updated: December 2024</p>
  <p><em>Comprehensive guide for Vivi_RAG_final3.py</em></p>
</div>
</style>
