import os
import uuid
import tempfile
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import whisper
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import re
import requests
from rag_pipeline import VideoRAG
import cv2
import json

# Import Google Drive sync
try:
    from google_drive_sync import GoogleDriveSync
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("⚠️ Google Drive sync not available. Install required packages: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# ------------------ Utility Functions ------------------
def _parse_srt_time(t):
    try:
        h, m, s_ms = t.split(":")
        if "," in s_ms:
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        else:
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
    except:
        pass
    try:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    except:
        pass
    try:
        return float(t)
    except:
        raise ValueError(f"Unrecognized timestamp format: {t}")

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

# ------------------ FFmpeg Functions ------------------
def concatenate_videos(video_paths, output_filepath):
    video_paths = [p for p in video_paths if p and os.path.exists(p)]
    if not video_paths:
        return "❌ Error: No valid video clips provided.", None
    try:
        if not video_paths:
            return "❌ Error: No video clips provided.", None
        temp_dir = tempfile.gettempdir()
        list_file_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in video_paths:
                normalized_path = path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        # Use re-encoding instead of stream copying for better compatibility
        # This ensures all clips have consistent codecs, frame rates, and audio settings
        command = [
            "ffmpeg", "-y", 
            "-f", "concat", 
            "-safe", "0",
            "-i", list_file_path, 
            "-c:v", "libx264",           # Use H.264 codec for video
            "-c:a", "aac",               # Use AAC codec for audio
            "-preset", "fast",           # Fast encoding preset
            "-crf", "23",                # Good quality setting
            "-r", "30",                  # Force 30fps for consistency
            "-ar", "44100",              # Force 44.1kHz audio sample rate
            "-ac", "2",                  # Force stereo audio
            "-movflags", "+faststart",   # Optimize for streaming
            "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
            "-fflags", "+genpts",        # Generate presentation timestamps
            output_filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Clean up temporary files
        try:
            os.remove(list_file_path)
            # Also clean up individual clip files
            for clip_path in video_paths:
                if os.path.exists(clip_path) and "clip_" in os.path.basename(clip_path):
                    os.remove(clip_path)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
        
        return f"✅ Concatenated video saved to: {output_filepath}", output_filepath
    except subprocess.CalledProcessError as e:
        return f"❌ FFmpeg error: {e.stderr}", None
    except Exception as e:
        return f"❌ Exception during concatenation: {str(e)}", None

# ------------------ LLM Setup ------------------
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
llm_clipping = ChatGroq(model="llama-3.3-70b-versatile", temperature=1.2)  # Higher temperature for more creative clipping

general_template = """
You are Vivi, an expert and friendly video assistant chatbot.
You are having an ongoing conversation with the user. You have access to a transcript of a video. If the user's question is about the video, answer helpfully and use the transcript as context. Do not refer to timestamps unless they are explicitly present in the transcript context. If the question is general and not related to the video, just respond helpfully like a normal assistant.
---
Conversation History:
{context}
---
Transcript of the Video:
{transcript}
---
User:
{question}
---
Vivi:
"""

clipping_template = """
You are an expert video analysis assistant. Given a user query and a set of relevant transcript segments (with timestamps), propose the best, smoothest video clip ranges (start and end times in seconds) that comprehensively answer the query. 

IMPORTANT GUIDELINES:
1. Create COMPLETE, comprehensive clips that fully address the query
2. For acronyms, frameworks, methods, or multi-part explanations, ensure ALL components are explained in the clip
3. Merge adjacent or nearby segments to create longer, more natural clips
4. Extend clips to include complete thoughts, sentences, and explanations
5. Ensure clips start and end at natural points (beginning/end of sentences or thoughts)
6. Do NOT create very short clips - aim for meaningful, substantial content
7. If the query asks for multiple aspects, create multiple clips to cover all aspects
8. Use the provided segments as context but feel free to extend beyond them for completeness
9. Pay special attention to structured content - make sure all parts are covered
10. AVOID creating overlapping or redundant clips - each clip should be unique and non-overlapping
11. Choose the MOST RELEVANT clip range that best answers the query
12. For acronym explanations, ensure you include the COMPLETE explanation of ALL letters/components
13. Do NOT cut off acronym explanations mid-way - include the full breakdown
14. If an acronym has multiple parts, make sure ALL parts are covered in the clip
15. If the query asks for MULTIPLE acronyms (e.g., "What is NOPP and StarULIP?"), create separate clips for EACH acronym
16. Ensure ALL requested acronyms are covered - don't miss any acronym mentioned in the query
17. CRITICAL: Do NOT create overlapping time ranges - each clip should have unique start and end times
18. For multiple acronyms, create one clip per acronym with non-overlapping time ranges
19. If segments overlap, choose the most relevant non-overlapping ranges

Query: {query}
Relevant Transcript Segments:
{transcript}

Return the result in the following format, with each range and explanation on separate lines:
- Range: start_time - end_time
  Relevance: [Brief explanation of why this range is relevant to the query]

Make sure to create comprehensive clips that fully answer the user's query, not just partial explanations. For structured content (acronyms, frameworks, methods), ensure every component is explained. Avoid redundant or overlapping clips. For acronyms, make sure to include the complete explanation of all components. If multiple acronyms are requested, ensure ALL are covered with separate, non-overlapping clips.
"""

prompt_general = ChatPromptTemplate.from_template(general_template)
prompt_clipping = ChatPromptTemplate.from_template(clipping_template)
chain_general = prompt_general | llm
chain_clipping = prompt_clipping | llm_clipping

# ------------------ Vivi GUI Class ------------------
class ViviChatbot:
    def __init__(self):
        # Use light appearance mode
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Create root and set background
        self.root = ctk.CTk()
        self.root.configure(fg_color="#FFFFFF")
        self.root.title("ClipQuery- Video Chatbot")
        self.root.geometry("900x650")

        self.video_path = ""
        self.context = ""
        self.transcript_segments = []
        self.full_transcript_text = ""
        self.audio_process = None

        # Initialize Google Drive sync in background
        self.drive_sync = None
        self.sync_thread = None
        if GOOGLE_DRIVE_AVAILABLE:
            self.init_google_drive_sync()

        # Logo and Title Frame
        self.header_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF", width=200, height=100)
        self.header_frame.pack(anchor="nw", padx=20, pady=(10, 5), fill="x")

        # Load and display logo with resizing
        try:
            logo_img = Image.open(r"C:\Users\adibr\Desktop\TSS\vivi_logo_1.PNG")
            logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_img)
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                image=logo_photo,
                text="",
                fg_color="#FFFFFF"
            )
            self.logo_label.image = logo_photo
            self.logo_label.pack(side="left", padx=(0, 10))
        except Exception as e:
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                text="Logo",
                font=("Arial", 16),
                fg_color="#FFFFFF",
                text_color="#1a2238"
            )
            self.logo_label.pack(side="left", padx=(0, 10))

        # Title label
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="ClipQuery",
            font=("Arial", 24, "bold"),
            text_color="#1a2238"
        )
        self.title_label.pack(side="left")

        # Chat display frame
        self.chat_frame = ctk.CTkScrollableFrame(self.root, width=880, height=500, fg_color="#FFFFFF")
        self.chat_frame.pack(padx=20, pady=(5, 10), fill="both", expand=True)
        self.scrollable_frame = self.chat_frame

        # Entry field frame
        self.entry_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF")
        self.entry_frame.pack(pady=(0, 10), fill="x", padx=20)

        self.user_entry = ctk.CTkEntry(self.entry_frame, placeholder_text="Ask Vivi...", width=700, height=40, font=("Arial", 16))
        self.user_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", lambda e: self.send_message())

        # Send button
        self.send_btn = ctk.CTkButton(self.entry_frame, text="Send", command=self.send_message, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.send_btn.pack(side="right")

        # Buttons for upload, input video, and final video
        self.btn_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF")
        self.btn_frame.pack(pady=(0, 10))

        self.upload_btn = ctk.CTkButton(self.btn_frame, text="📂 Upload Video", command=self.browse_video, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.upload_btn.pack(side="left", padx=5)

        self.input_video_btn = ctk.CTkButton(self.btn_frame, text="▶️ Input Video", command=self.play_input_video, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.input_video_btn.pack(side="left", padx=5)

        self.final_video_btn = ctk.CTkButton(self.btn_frame, text="🎬 Final Video", command=self.play_final_video, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.final_video_btn.pack(side="left", padx=5)

        # Sync button
        self.sync_btn = ctk.CTkButton(self.btn_frame, text="🔄 Sync Drive", command=self.manual_sync, fg_color="#17a2b8", width=120, height=40, font=("Arial", 14))
        self.sync_btn.pack(side="left", padx=5)

        # Google Drive sync status indicator
        self.sync_status_label = ctk.CTkLabel(self.btn_frame, text="🔄 Initializing Google Drive sync...", font=("Arial", 12), text_color="#666666")
        self.sync_status_label.pack(side="right", padx=10)

        # Progress bar for buffering
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.root, variable=self.progress_var, width=200)
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # Hide initially

        # Initial welcome message
        self.display_message("assistant", "Hello! I'm Vivi, your video assistant. I can help you with video analysis, transcription, and creating clips. How can I assist you today?")

        # Initialize RAG pipeline
        try:
            self.rag = VideoRAG()
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
            self.rag = None

        # Similarity threshold for filtering results
        self.similarity_threshold = 0.85

    def init_google_drive_sync(self):
        """Initialize Google Drive sync in background thread."""
        def sync_worker():
            try:
                print("🔄 Initializing Google Drive sync...")
                self.drive_sync = GoogleDriveSync()
                
                # Perform initial sync
                print("📥 Performing initial sync from Google Drive...")
                sync_result = self.drive_sync.initial_sync()
                
                print(f"✅ Google Drive sync initialized successfully")
                print(f"📥 Downloaded: {len(sync_result['downloaded'])} files")
                print(f"🔄 Updated: {len(sync_result['updated'])} files")
                
                # Update status label safely
                try:
                    self.root.after(0, lambda: self.sync_status_label.configure(
                        text="✅ Google Drive sync active",
                        text_color="#28a745"
                    ))
                except Exception as e:
                    print(f"⚠️ Could not update status label: {e}")
                
                # Start watching for changes
                self.drive_sync.start_watching()
                print("👀 Google Drive sync is now running in background")
                
            except Exception as e:
                print(f"❌ Error initializing Google Drive sync: {e}")
                self.drive_sync = None
                # Update status label to show error safely
                try:
                    self.root.after(0, lambda: self.sync_status_label.configure(
                        text="❌ Google Drive sync failed",
                        text_color="#dc3545"
                    ))
                    # Show error message in chat
                    self.root.after(0, lambda: self.display_message("assistant", 
                        f"⚠️ Google Drive sync is not available: {str(e)}\n\n"
                        "The chatbot will work with local files only. "
                        "To enable Google Drive sync, please check your credentials and try again."
                    ))
                except Exception as ui_error:
                    print(f"⚠️ Could not update UI: {ui_error}")
        
        # Start sync in background thread
        self.sync_thread = threading.Thread(target=sync_worker, daemon=True)
        self.sync_thread.start()

    def show_buffering(self):
        self.buffering_label = ctk.CTkLabel(self.root, text="Analyzing", font=("Arial", 16))
        self.buffering_label.pack(pady=10)
        self.animate_buffering(0)

    def animate_buffering(self, count):
        dots = "." * (count % 4)
        self.buffering_label.configure(text=f"Analyzing{dots}")
        self.root.after(500, lambda: self.animate_buffering(count + 1))

    def hide_buffering(self):
        self.buffering_label.pack_forget()
        
    def typewriter_effect(self, sender, message):
        outer_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#FFFFFF")
        outer_frame.pack(fill="x", padx=10, pady=5)

        bubble_frame = ctk.CTkFrame(
            outer_frame,
            fg_color="#3a0e2e" if sender.lower() == "vivi" else "#1B263B",
            corner_radius=12
        )
        if sender.lower() == "vivi":
            bubble_frame.pack(anchor="w", padx=5)
        else:
            bubble_frame.pack(anchor="e", padx=5)

        name_label = ctk.CTkLabel(
            bubble_frame,
            text=sender + ":",
            font=("Arial", 12, "bold"),
            text_color="#F0F0F0"
        )
        name_label.pack(anchor="w", padx=8, pady=(6, 0))

        message_label = ctk.CTkLabel(
            bubble_frame,
            text="",
            font=("Arial", 20, "normal"),
            wraplength=800,
            justify="left",
            text_color="#F0F0F0"
        )
        message_label.pack(anchor="w", padx=8, pady=(0, 8))

        self.root.update_idletasks()
        self.chat_frame._parent_canvas.yview_moveto(1.0)

        current_text = ""
        for word in message.split():
            current_text += word + " "
            message_label.configure(text=current_text)
            self.root.update_idletasks()
            self.chat_frame._parent_canvas.yview_moveto(1.0)
            time.sleep(0.04)

    def display_message(self, sender, message):
        outer_frame = ctk.CTkFrame(self.chat_frame, fg_color="#FFFFFF")
        outer_frame.pack(fill="x", padx=10, pady=5)

        bubble = ctk.CTkFrame(
            outer_frame,
            corner_radius=15,
            fg_color="#3a0e2e" if sender.lower() != "user" else "#1a2238"
        )

        if sender.lower() == "user":
            bubble.pack(anchor="e", padx=5)
        else:
            bubble.pack(anchor="w", padx=5)

        label = ctk.CTkLabel(
            bubble,
            text=f"{message}",
            wraplength=800,
            font=("Arial", 20, "normal"),
            justify="left",
            text_color="#F0F0F0"
        )
        label.pack(anchor="w", padx=10, pady=5)

    def clip_video(self, video_path, start_time, end_time):
        if not video_path or not os.path.exists(video_path):
            self.display_message("system", "❌ No valid video found for clipping.")
            return "No video found", None
        try:
            if start_time >= end_time:
                self.display_message("system", f"⚠ Invalid clip range: {start_time} >= {end_time}")
                return "Invalid time range", None
            
            clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), clip_filename)
            
            # Use re-encoding instead of stream copying to prevent frame freezing and sync issues
            # Force keyframe alignment and consistent codec settings
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(end_time - start_time),
                "-c:v", "libx264",           # Use H.264 codec for video
                "-c:a", "aac",               # Use AAC codec for audio
                "-preset", "fast",           # Fast encoding preset
                "-crf", "23",                # Good quality setting
                "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
                "-fflags", "+genpts",        # Generate presentation timestamps
                "-r", "30",                  # Force 30fps for consistency
                "-ar", "44100",              # Force 44.1kHz audio sample rate
                "-ac", "2",                  # Force stereo audio
                "-movflags", "+faststart",   # Optimize for streaming
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return f"🎬 Video clip saved to {output_path}", output_path
        except subprocess.CalledProcessError as e:
            self.display_message("system", f"❌ FFmpeg error: {e.stderr}")
            return f"FFmpeg error: {e.stderr}", None
        except Exception as e:
            self.display_message("system", f"❌ Error clipping video: {e}")
            return f"Error clipping video: {e}", None

    def transcribe_video(self, video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        dir_ = os.path.dirname(video_path)
        txt_path = os.path.join(dir_, f"{base}.txt")
        srt_path = os.path.join(dir_, f"{base}.srt")
        if os.path.exists(txt_path) and os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                blocks = f.read().strip().split("\n\n")
            with open(txt_path, "r", encoding="utf-8") as f:
                timed_transcript = f.read().strip()
            segments = []
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 3:
                    times = lines[1].split(" --> ")
                    start = _parse_srt_time(times[0])
                    end = _parse_srt_time(times[1])
                    text = " ".join(lines[2:])
                    segments.append({"start": start, "end": end, "text": text})
            return timed_transcript, segments
        self.display_message("system", "🔍 Running Whisper transcription...")
        
        self.show_buffering()
        model = whisper.load_model("medium")
        
        
        result = model.transcribe(video_path, verbose=True)
        segments = []
        timed_transcript = ""
        for seg in result['segments']:
            segments.append({"start": seg['start'], "end": seg['end'], "text": seg['text']})
            timed_transcript += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(timed_transcript)
        self.hide_buffering()
        return timed_transcript, segments
    
    # ################## Using API #################
    # def transcribe_video(self, video_path):
    #     base = os.path.splitext(os.path.basename(video_path))[0]
    #     dir_ = os.path.dirname(video_path)
    #     txt_path = os.path.join(dir_, f"{base}.txt")
    #     srt_path = os.path.join(dir_, f"{base}.srt")

    #     # Check if already transcribed
    #     if os.path.exists(txt_path) and os.path.exists(srt_path):
    #         with open(srt_path, "r", encoding="utf-8") as f:
    #             blocks = f.read().strip().split("\n\n")
    #         with open(txt_path, "r", encoding="utf-8") as f:
    #             timed_transcript = f.read().strip()
    #         segments = []
    #         for block in blocks:
    #             lines = block.split("\n")
    #             if len(lines) >= 3:
    #                 times = lines[1].split(" --> ")
    #                 start = _parse_srt_time(times[0])
    #                 end = _parse_srt_time(times[1])
    #                 text = " ".join(lines[2:])
    #                 segments.append({"start": start, "end": end, "text": text})
    #         return timed_transcript, segments

    #     # Use Groq Whisper API
    #     self.display_message("system", "🔍 Sending video to Groq Whisper (large-v3-turbo) for transcription...")

    #     api_key = os.environ.get("GROQ_API_KEY")
    #     if not api_key:
    #         raise ValueError("GROQ_API_KEY is not set in environment variables.")

    #     client = Groq(api_key=api_key)
    #     try:
    #         with open(video_path, "rb") as file:
    #             transcription = client.audio.transcriptions.create(
    #                 file=file,
    #                 model="whisper-large-v3-turbo",
    #                 prompt="",  # You can provide custom prompt if needed
    #                 response_format="verbose_json",
    #                 timestamp_granularities=["segment"],
    #                 language="en",
    #                 temperature=0.0
    #             )

    #         segments = []
    #         timed_transcript = ""
    #         for seg in transcription.segments:
    #             start = seg["start"]
    #             end = seg["end"]
    #             text = seg["text"]
    #             segments.append({"start": start, "end": end, "text": text})
    #             timed_transcript += f"[{start:.2f} - {end:.2f}] {text}\n"

    #         # Save to SRT
    #         with open(srt_path, "w", encoding="utf-8") as f:
    #             for i, seg in enumerate(segments):
    #                 f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")

    #         # Save to TXT
    #         with open(txt_path, "w", encoding="utf-8") as f:
    #             f.write(timed_transcript)

    #         return timed_transcript, segments

    #     except Exception as e:
    #         self.display_message("system", f"Transcription Failed: {str(e)}")
    #         return "", []

    def load_transcript_segments(self, video_id, relevant_ranges=None):
        """Load transcript segments for a video. If relevant_ranges is provided, only load those segments."""
        txt_path = os.path.join("Max Life Videos", f"{video_id}.txt")
        txt_path = os.path.normpath(txt_path)
        segments = []
        if not os.path.exists(txt_path):
            print(f"Transcript file not found: {txt_path}")
            return segments
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(.+)', line)
                    if match:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        text = match.group(3).strip()
                        if relevant_ranges:
                            for r_start, r_end in relevant_ranges:
                                if (start >= r_start and start < r_end) or (end > r_start and end <= r_end):
                                    segments.append({"start": start, "end": end, "text": text})
                                    break
                        else:
                            segments.append({"start": start, "end": end, "text": text})
        except Exception as e:
            print(f"Error reading transcript file {txt_path}: {e}")
        return segments

    def load_full_transcript(self, video_id):
        """Load the full transcript for a video."""
        txt_path = os.path.join("Max Life Videos", f"{video_id}.txt")
        txt_path = os.path.normpath(txt_path)
        if not os.path.exists(txt_path):
            print(f"Transcript file not found: {txt_path}")
            return ""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading transcript file {txt_path}: {e}")
            return ""

    def debug_video_usage(self, rag_results, query_type="normal"):
        """Debug helper to show which video parts are being used."""
        if not rag_results:
            print(f"=== {query_type.upper()} QUERY: No RAG results found ===")
            return
        
        video_ids = list({r['video_id'] for r in rag_results})
        print(f"=== {query_type.upper()} QUERY DEBUG ===")
        print(f"Total videos found: {len(video_ids)}")
        print(f"Video IDs: {video_ids}")
        print(f"Total segments: {len(rag_results)}")
        
        # Group segments by video
        segments_by_video = {}
        for result in rag_results:
            vid = result['video_id']
            if vid not in segments_by_video:
                segments_by_video[vid] = []
            segments_by_video[vid].append(result)
        
        print(f"\n=== SEGMENTS BY VIDEO ===")
        for video_id, segments in segments_by_video.items():
            print(f"\nVideo {video_id}:")
            print(f"  Segments: {len(segments)}")
            print(f"  Time range: {min(s['start'] for s in segments):.2f}s - {max(s['end'] for s in segments):.2f}s")
            print(f"  Total duration: {max(s['end'] for s in segments) - min(s['start'] for s in segments):.2f}s")
            
            # Show first few segments
            for i, seg in enumerate(segments[:3]):
                print(f"    Segment {i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s")
                print(f"      Text: {seg['text'][:80]}...")
                print(f"      Similarity: {seg['similarity']:.3f}")
            
            if len(segments) > 3:
                print(f"    ... and {len(segments) - 3} more segments")

    def find_natural_ending(self, video_id, end_time, look_back=10):
        """Find a natural ending point within the look_back range, or extend to the next sentence-ending punctuation or farewell."""
        full_segments = self.load_transcript_segments(video_id)
        if not full_segments:
            return end_time

        # Expanded set of farewell/closure keywords
        closure_keywords = [
            'thank you', "that's all", 'in conclusion', 'finally', 'summary',
            'goodbye', 'see you', 'take care', 'have a great day', 'farewell',
            'bye', 'see you next time', 'until next time', 'hope this helps',
            'let me know if you have questions', 'concludes', 'conclusion', 'wrapping up', 'wrap up', 'end of', 'ending', 'all the best', 'best wishes', 'wish you', 'good night', 'good morning', 'good afternoon', 'good evening'
        ]
        # 1. Look for segments that end near the proposed end time and end naturally
        natural_endings = []
        for i, seg in enumerate(full_segments):
            if abs(seg['end'] - end_time) <= look_back:
                text = seg['text'].strip()
                if text.endswith(('.', '!', '?', ':', ';')) or any(keyword in text.lower() for keyword in closure_keywords):
                    # Also prefer if this is the last segment or a long pause follows
                    is_last = (i == len(full_segments) - 1)
                    next_start = full_segments[i+1]['start'] if i+1 < len(full_segments) else None
                    pause_after = (next_start - seg['end']) if next_start is not None else None
                    natural_endings.append((seg['end'], is_last, pause_after if pause_after is not None else 0))
        if natural_endings:
            # Prefer last segment, then longer pause, then closest
            natural_endings.sort(key=lambda x: (not x[1], -x[2], abs(x[0] - end_time)))
            chosen = natural_endings[0][0]
            print(f"Found natural ending at {chosen:.2f}s (was {end_time:.2f}s)")
            return chosen

        # 2. If no good ending is found, extend to the next sentence-ending punctuation after end_time
        for seg in full_segments:
            if seg['start'] >= end_time:
                text = seg['text'].strip()
                if text.endswith(('.', '!', '?', ':', ';')) or any(keyword in text.lower() for keyword in closure_keywords):
                    print(f"Extended to next natural ending at {seg['end']:.2f}s (was {end_time:.2f}s)")
                    return seg['end']

        # 3. If the clip is near the end of the video, prefer the last segment
        if full_segments and end_time > full_segments[-1]['end'] - 5:
            print(f"Clip is near the end, using last segment end at {full_segments[-1]['end']:.2f}s")
            return full_segments[-1]['end']

        # 4. Otherwise, return the original end_time
        return end_time

    def find_complete_acronym_explanation(self, video_id, query):
        """Find complete acronym explanations in a video for any acronym query."""
        full_segments = self.load_transcript_segments(video_id)
        
        # Extract only actual acronyms from query (not common words)
        query_upper = query.upper()
        # Look for specific known acronyms first
        known_acronyms = ['NOPP', 'STARULIP', 'ULIP', 'SMART', 'SWOT', 'ABCD', 'ABCDEF']
        acronyms = [acronym for acronym in known_acronyms if acronym in query_upper]
        
        # If no known acronyms found, look for general acronym patterns (3+ letters)
        if not acronyms:
            general_acronyms = re.findall(r'\b[A-Z]{3,}\b', query_upper)
            # Filter out common words that might be mistaken for acronyms
            common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY'}
            acronyms = [acronym for acronym in general_acronyms if acronym not in common_words]
        
        print(f"Looking for acronyms: {acronyms}")
        
        # Look for segments that contain acronym-related content
        acronym_segments = []
        for seg in full_segments:
            text = seg['text'].lower()
            
            # Check for acronym patterns in the text
            text_acronyms = re.findall(r'\b[A-Z]{2,}\b', seg['text'].upper())
            
            # Check for acronym-related keywords
            acronym_keywords = [
                'criteria', 'need', 'opportunity', 'physically', 'paying', 'capacity',
                'specific', 'measurable', 'achievable', 'relevant', 'time-bound',
                'strengths', 'weaknesses', 'opportunities', 'threats',
                'analysis', 'framework', 'method', 'approach', 'strategy',
                'ulip', 'unit linked', 'investment', 'policy', 'charges', 'premium',
                'allocation', 'mortality', 'admin', 'returns', 'benefits'
            ]
            
            has_acronym = any(acronym in text_acronyms for acronym in acronyms)
            has_keywords = any(keyword in text for keyword in acronym_keywords)
            
            if has_acronym or has_keywords:
                acronym_segments.append(seg)
        
        if len(acronym_segments) >= 2:  # Should have at least 2 segments for an acronym
            # Sort by time and merge
            sorted_segments = sorted(acronym_segments, key=lambda x: x['start'])
            start_time = sorted_segments[0]['start']
            end_time = sorted_segments[-1]['end']
            
            # For acronyms, we want to ensure we get the complete explanation
            # Look for additional segments that might contain related content
            full_segments = self.load_transcript_segments(video_id)
            
            # Find segments that are close to our acronym segments and might contain related content
            expanded_segments = []
            for seg in full_segments:
                # Check if this segment is close to our acronym segments
                is_close = any(abs(seg['start'] - s['start']) <= 30 for s in sorted_segments)
                
                # Check if it contains related keywords
                text = seg['text'].lower()
                related_keywords = [
                    'criteria', 'need', 'opportunity', 'physically', 'paying', 'capacity',
                    'specific', 'measurable', 'achievable', 'relevant', 'time-bound',
                    'strengths', 'weaknesses', 'opportunities', 'threats',
                    'analysis', 'framework', 'method', 'approach', 'strategy',
                    'first', 'second', 'third', 'fourth', 'finally',
                    'step', 'phase', 'stage', 'level', 'tier',
                    'benefit', 'advantage', 'feature', 'characteristic', 'property',
                    'example', 'instance', 'case', 'scenario', 'situation'
                ]
                
                has_related = any(keyword in text for keyword in related_keywords)
                
                if is_close or has_related:
                    expanded_segments.append(seg)
            
            # Merge all related segments
            if expanded_segments:
                # Use a more robust way to merge segments without duplicates
                all_segments = sorted_segments.copy()
                for exp_seg in expanded_segments:
                    # Check if this segment is already in our list
                    is_duplicate = any(
                        abs(exp_seg['start'] - seg['start']) < 1.0 and 
                        abs(exp_seg['end'] - seg['end']) < 1.0 
                        for seg in all_segments
                    )
                    if not is_duplicate:
                        all_segments.append(exp_seg)
                
                # Sort by start time
                all_segments = sorted(all_segments, key=lambda x: x['start'])
                start_time = all_segments[0]['start']
                end_time = all_segments[-1]['end']
                print(f"Expanded acronym explanation from {len(sorted_segments)} to {len(all_segments)} segments")
            else:
                all_segments = sorted_segments
            
            # Limit the duration to avoid overly long clips (max 60 seconds for acronyms)
            max_duration = 60.0  # Increased from 40.0
            if end_time - start_time > max_duration:
                # Keep the most relevant part (center of the explanation)
                center = (start_time + end_time) / 2
                start_time = max(0, center - max_duration / 2)
                end_time = min(start_time + max_duration, end_time)
            
            # Get only the segments within the limited range
            limited_segments = [seg for seg in all_segments if seg['start'] >= start_time and seg['end'] <= end_time]
            full_text = ' '.join([s['text'] for s in limited_segments])
            
            # Validate that the acronym explanation is complete
            if not self.validate_acronym_completeness(full_text, acronyms):
                print("Warning: Acronym explanation may be incomplete, trying to expand...")
                # Try to expand the range to get more complete explanation
                expansion = 20.0  # Add 20 seconds on each side
                start_time = max(0, start_time - expansion)
                end_time = end_time + expansion
                
                # Get expanded segments
                expanded_limited = [seg for seg in all_segments if seg['start'] >= start_time and seg['end'] <= end_time]
                full_text = ' '.join([s['text'] for s in expanded_limited])
                
                # Re-validate
                if not self.validate_acronym_completeness(full_text, acronyms):
                    print("Warning: Still incomplete, but proceeding with available content")
            
            # Calculate relevance score (lower is better)
            relevance_score = 0.5  # Default score
            if acronyms:
                # Check how many acronyms are mentioned in the text
                mentioned_acronyms = sum(1 for acronym in acronyms if acronym in full_text.upper())
                relevance_score = 1.0 - (mentioned_acronyms / len(acronyms))
            
            print(f"Found complete acronym explanation: {start_time:.2f}-{end_time:.2f} (duration: {end_time-start_time:.2f}s, relevance: {relevance_score:.3f})")
            print(f"Text preview: {full_text[:100]}...")
            
            # Store video_id information for each acronym segment to help with clip matching
            for seg in limited_segments:
                seg['video_id'] = seg.get('video_id', '')
                print(f"Acronym segment {seg['start']:.2f}-{seg['end']:.2f} from video {seg['video_id']}")
            
            return {
                'start': start_time,
                'end': end_time,
                'text': full_text,
                'video_id': video_id,
                'similarity': relevance_score
            }
        
        return None

    def find_complete_nopp_explanation(self, video_id):
        """Specifically find the complete NOPP explanation in a video."""
        full_segments = self.load_transcript_segments(video_id)
        
        # Look for segments that contain NOPP-related content
        nopp_segments = []
        for seg in full_segments:
            text = seg['text'].lower()
            if any(keyword in text for keyword in ['nop', 'criteria', 'need for insurance', 'opportunity to meet', 'physically fit', 'paying capacity']):
                nopp_segments.append(seg)
        
        if len(nopp_segments) >= 4:  # Should have at least 4 segments for NOPP
            # Sort by time and merge
            sorted_segments = sorted(nopp_segments, key=lambda x: x['start'])
            start_time = sorted_segments[0]['start']
            end_time = sorted_segments[-1]['end']
            full_text = ' '.join([s['text'] for s in sorted_segments])
            
            print(f"Found complete NOPP explanation: {start_time:.2f}-{end_time:.2f}")
            return {
                'start': start_time,
                'end': end_time,
                'text': full_text,
                'video_id': video_id
            }
        
        return None

    def expand_segments_for_completeness(self, segments, max_expansion=60):
        """Expand segments to ensure complete topic coverage, especially for acronyms and multi-part explanations."""
        if not segments:
            return segments
        
        expanded_segments = []
        
        for seg in segments:
            # Start with the original segment
            expanded = seg.copy()
            
            # Look for potential acronyms, multi-part explanations, or structured content
            text = seg['text'].lower()
            
            # Keywords that indicate multi-part explanations
            multi_part_keywords = [
                'criteria', 'need', 'opportunity', 'physically', 'paying', 'capacity',
                'specific', 'measurable', 'achievable', 'relevant', 'time-bound',
                'strengths', 'weaknesses', 'opportunities', 'threats',
                'analysis', 'framework', 'method', 'approach', 'strategy',
                'first', 'second', 'third', 'fourth', 'finally',
                'step', 'phase', 'stage', 'level', 'tier',
                'benefit', 'advantage', 'feature', 'characteristic', 'property'
            ]
            
            # Check for acronym patterns
            acronym_patterns = re.findall(r'\b[A-Z]{2,}\b', seg['text'].upper())
            
            has_multi_part = any(keyword in text for keyword in multi_part_keywords)
            has_acronym = len(acronym_patterns) > 0
            
            if has_multi_part or has_acronym:
                print(f"Potential multi-part segment found: {seg['text'][:50]}...")
                
                # Load the full transcript for this video to find all related segments
                video_id = seg['video_id']
                full_segments = self.load_transcript_segments(video_id)
                
                # Find all segments within the expansion range that might be related
                related_segments = []
                for full_seg in full_segments:
                    if abs(full_seg['start'] - seg['start']) <= max_expansion:
                        # Check if this segment contains related content
                        full_text = full_seg['text'].lower()
                        if any(keyword in full_text for keyword in multi_part_keywords):
                            related_segments.append(full_seg)
                
                if len(related_segments) > 1:
                    # Merge related segments
                    sorted_related = sorted(related_segments, key=lambda x: x['start'])
                    expanded['start'] = sorted_related[0]['start']
                    expanded['end'] = sorted_related[-1]['end']
                    expanded['text'] = ' '.join([s['text'] for s in sorted_related])
                    print(f"Expanded segment: {expanded['start']:.2f}-{expanded['end']:.2f}")
                    print(f"Expanded text preview: {expanded['text'][:100]}...")
            
            expanded_segments.append(expanded)
        
        return expanded_segments

    def filter_relevant_segments(self, segments, max_segments=8):
        """Filter and rank segments by relevance for normal queries."""
        if not segments:
            return segments
        
        # Sort by similarity (lower is better)
        sorted_segments = sorted(segments, key=lambda x: x['similarity'])
        
        # Take the most relevant segments
        filtered = sorted_segments[:max_segments]
        
        # Additional filtering: remove segments that are too similar to each other
        # But be less aggressive to ensure we don't miss important content
        unique_segments = []
        for seg in filtered:
            # Check if this segment is too similar to already selected segments
            is_duplicate = False
            for existing in unique_segments:
                # If segments are from same video and close in time, consider them similar
                # But use a larger time window to avoid missing related content
                if (seg['video_id'] == existing['video_id'] and 
                    abs(seg['start'] - existing['start']) < 15):  # Reduced from 30 seconds
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_segments.append(seg)
        
        # If we filtered out too many segments, add some back
        if len(unique_segments) < max_segments // 2:
            print(f"Warning: Filtered too aggressively, adding back segments")
            for seg in filtered:
                # Check if this segment is already in unique_segments by comparing key properties
                already_included = any(
                    seg['video_id'] == existing['video_id'] and 
                    abs(seg['start'] - existing['start']) < 1.0 and
                    abs(seg['end'] - existing['end']) < 1.0
                    for existing in unique_segments
                )
                if not already_included:
                    unique_segments.append(seg)
                    if len(unique_segments) >= max_segments:
                        break
        
        return unique_segments

    def validate_and_improve_clips(self, clips, video_duration, is_acronym_query=False):
        """Validate and improve clip ranges to ensure they are comprehensive."""
        improved_clips = []
        
        for clip in clips:
            start, end = clip['start'], clip['end']
            
            # Different duration limits for acronym vs normal queries
            if is_acronym_query:
                min_duration = 15.0  # Longer for acronyms to ensure complete coverage
                max_duration = 60.0  # Increased from 45.0 for complete acronym explanations
            else:
                min_duration = 8.0   # Shorter for normal queries
                max_duration = 25.0  # Reasonable limit for normal queries
            
            # Ensure minimum clip duration
            if end - start < min_duration:
                # Extend the clip to meet minimum duration
                extension = (min_duration - (end - start)) / 2
                start = max(0, start - extension)
                end = min(video_duration, end + extension)
            
            # Cap maximum duration
            if end - start > max_duration:
                # Reduce clip length while keeping the most relevant part
                center = (start + end) / 2
                start = max(0, center - max_duration / 2)
                end = min(video_duration, center + max_duration / 2)
            
            # Find natural ending for smoother clips (only for non-acronym queries)
            if not is_acronym_query:
                end = self.find_natural_ending(clip.get('video_id', ''), end, look_back=8)
            else:
                # For acronym queries, don't cut off early - let the full explanation play out
                print("Acronym query detected - preserving full explanation length")
            
            # Ensure clips don't exceed video duration
            if start >= video_duration:
                continue
            if end > video_duration:
                end = video_duration
            
            # Ensure start < end
            if start >= end:
                continue
            
            improved_clips.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'original': clip
            })
        
        # Sort by duration (longer clips first) and relevance
        improved_clips.sort(key=lambda x: x['duration'], reverse=True)
        
        return improved_clips

    def merge_adjacent_segments(self, segments, max_gap=5.0):
        """Merge adjacent segments that are close in time to create more comprehensive clips."""
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        merged = []
        current = sorted_segments[0].copy()
        
        for next_seg in sorted_segments[1:]:
            # If segments are close enough, merge them
            if next_seg['start'] - current['end'] <= max_gap:
                current['end'] = next_seg['end']
                current['text'] += ' ' + next_seg['text']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged

    def build_llm_context(self, query, for_clipping=False):
        """Use RAG to find relevant videos/segments and build the transcript context for the LLM."""
        if not self.rag:
            return "", []

        # --- 1. Extract all unique acronyms/rare capitalized terms from the query ---
        query_upper = query.upper()
        # Known acronyms and rare terms (extendable)
        known_acronyms = ['NOPP', 'STARULIP', 'ULIP', 'SMART', 'SWOT', 'ABCD', 'ABCDEF']
        # Extract known acronyms
        detected_acronyms = [acronym for acronym in known_acronyms if acronym in query_upper]
        # Also extract any 3+ letter capitalized words not in common words
        general_acronyms = re.findall(r'\b[A-Z]{3,}\b', query_upper)
        common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY', 'CLIPPING', 'QUERY', 'VIDEO', 'CHAT', 'BOT', 'ASSISTANT', 'EXPLAIN', 'DESCRIBE', 'TELL', 'ABOUT', 'MEANING', 'DEFINITION'}
        detected_acronyms += [a for a in general_acronyms if a not in common_words and a not in detected_acronyms]
        detected_acronyms = list(set(detected_acronyms))  # Unique only

        # If no acronyms found, treat the whole query as a single topic
        if not detected_acronyms:
            subqueries = [query]
        else:
            # For each acronym, create a subquery ("What is <acronym>?")
            subqueries = [f"What is {acronym}?" for acronym in detected_acronyms]

        print(f"Multi-topic RAG: subqueries = {subqueries}")

        # --- 2. For each subquery, run a separate RAG search and merge results ---
        all_rag_results = []
        for subq in subqueries:
            n_results = 30 if for_clipping else 15
            rag_results = self.rag.query_videos(subq, n_results=n_results)
            all_rag_results.extend(rag_results)
        # Remove duplicates (by video_id, start, end)
        seen = set()
        unique_rag_results = []
        for r in all_rag_results:
            key = (r['video_id'], r['start'], r['end'])
            if key not in seen:
                unique_rag_results.append(r)
                seen.add(key)

        # Use a more inclusive similarity threshold for multi-topic queries
        similarity_threshold = 0.85 if for_clipping else 0.80
        filtered = [r for r in unique_rag_results if r['similarity'] <= similarity_threshold]
        video_ids = list({r['video_id'] for r in filtered})
        print(f"RAG search found {len(unique_rag_results)} unique results, filtered to {len(filtered)} from {len(video_ids)} videos")
        print(f"Video IDs found: {video_ids}")

        # --- 3. Context Construction (same as before) ---
        if len(video_ids) < 4:
            print(f"Using full transcripts for {len(video_ids)} videos")
            context_lines = []
            total_chars = 0
            max_total_chars = 24000 if for_clipping else 16000
            for video_id in video_ids:
                full_transcript = self.load_full_transcript(video_id)
                if full_transcript:
                    if len(full_transcript) > 6000:
                        full_transcript = full_transcript[:5997] + "..."
                    if total_chars + len(full_transcript) > max_total_chars:
                        break
                    context_lines.append(full_transcript)
                    total_chars += len(full_transcript)
            return "\n\n".join(context_lines), filtered
        else:
            print(f"Using segments for {len(video_ids)} videos")
            max_segments = 12 if for_clipping else 8
            filtered = filtered[:max_segments]
            if not for_clipping:
                filtered = self.filter_relevant_segments(filtered, max_segments=8)
                print(f"After relevance filtering: {len(filtered)} segments")
            additional_context = []
            if not for_clipping and (len(video_ids) <= 2 or self.is_definition_query(query)):
                print("Adding broader context for definition queries...")
                for video_id in video_ids:
                    full_transcript = self.load_full_transcript(video_id)
                    if full_transcript:
                        if len(full_transcript) > 4000:
                            full_transcript = full_transcript[:3997] + "..."
                        additional_context.append(full_transcript)
            max_total_chars = 24000 if for_clipping else 16000
            context_lines = []
            total_chars = 0
            for r in filtered:
                text = r['text']
                if len(text) > 600:
                    text = text[:597] + "..."
                if for_clipping:
                    line = f"[{r['start']:.2f} - {r['end']:.2f}] {text}"
                else:
                    line = text
                if total_chars + len(line) > max_total_chars:
                    break
                context_lines.append(line)
                total_chars += len(line)
            if additional_context:
                context_lines.extend(additional_context)
                print(f"Added {len(additional_context)} full transcripts for broader context")
            return "\n".join(context_lines), filtered

    def send_message(self):
        user_input = self.user_entry.get()
        if user_input.strip().lower() == "exit":
            self.root.destroy()
            return
        self.display_message("user", user_input)
        self.user_entry.delete(0, tk.END)
        self.user_entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        def extract_content(response):
            return str(response.content) if hasattr(response, "content") else str(response)

        def run_bot():
            if user_input.lower().startswith("clipping:"):
                query = user_input[len("clipping:"):].strip()
                context, rag_results = self.build_llm_context(query, for_clipping=True)
                
                # Debug: Show what context is being used for clipping
                print(f"=== CLIPPING DEBUG ===")
                print(f"Query: {query}")
                print(f"Context length: {len(context)}")
                print(f"RAG results count: {len(rag_results) if rag_results else 0}")
                
                # Use the debug helper
                self.debug_video_usage(rag_results, "clipping")
                
                # Get unique video IDs
                video_ids = list({r['video_id'] for r in rag_results}) if rag_results else []
                print(f"Unique video IDs: {video_ids}")
                
                # For clipping, we need to provide timestamped segments to the LLM
                # If we have full transcripts (< 4 videos), we need to extract segments from them
                if len(video_ids) < 4 and rag_results:
                    print("Using full transcripts for clipping - extracting segments")
                    
                    # Check if this is an acronym query and handle specially
                    query_lower = query.lower()
                    is_acronym_query = any(keyword in query_lower for keyword in ['acronym', 'meaning', 'what is', 'stands for']) and any(acronym in query.upper() for acronym in ['NOPP', 'ABCD', 'SMART', 'SWOT', 'ABCDEF', 'ULIP'])
                    
                    if is_acronym_query:
                        print("Acronym query detected - looking for complete explanation")
                        acronym_segments = []
                        
                        # Extract all acronyms from the query
                        query_upper = query.upper()
                        detected_acronyms = re.findall(r'\b[A-Z]{2,}\b', query_upper)
                        # Filter out common words
                        common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION'}
                        detected_acronyms = [a for a in detected_acronyms if a not in common_words]
                        print(f"Detected acronyms in query: {detected_acronyms}")
                        
                        # For each video, find explanations for the acronyms it contains
                        for video_id in video_ids:
                            complete_acronym = self.find_complete_acronym_explanation(video_id, query)
                            if complete_acronym:
                                # Check if this video contains any of the requested acronyms
                                video_text = complete_acronym['text'].upper()
                                contains_requested = any(acronym in video_text for acronym in detected_acronyms)
                                
                                if contains_requested:
                                    acronym_segments.append(complete_acronym)
                                    print(f"Added {video_id} explanation for acronyms: {[a for a in detected_acronyms if a in video_text]}")
                        
                        if acronym_segments:
                            print(f"Found {len(acronym_segments)} complete acronym explanations")
                            
                            # If we have multiple acronym explanations, keep all relevant ones
                            if len(acronym_segments) > 1:
                                print("Multiple acronym explanations found, keeping all relevant ones...")
                                # Don't filter out - keep all explanations for different acronyms
                                print(f"Keeping all {len(acronym_segments)} explanations")
                            
                            context_lines = [f"[{r['start']:.2f} - {r['end']:.2f}] {r['text']}" for r in acronym_segments]
                            rag_context = "\n".join(context_lines)
                            
                            # Store video_id information for each acronym segment to help with clip matching
                            for seg in acronym_segments:
                                seg['video_id'] = seg.get('video_id', '')
                                print(f"Acronym segment {seg['start']:.2f}-{seg['end']:.2f} from video {seg['video_id']}")
                        else:
                            # Fallback to normal processing
                            sorted_rag = sorted(rag_results, key=lambda r: r['similarity'], reverse=True)
                            merged_segments = self.merge_adjacent_segments(sorted_rag, max_gap=3.0)
                            expanded_segments = self.expand_segments_for_completeness(merged_segments, max_expansion=60)
                            context_lines = [f"[{r['start']:.2f} - {r['end']:.2f}] {r['text']}" for r in expanded_segments]
                            rag_context = "\n".join(context_lines)
                    else:
                        # Normal processing for non-acronym queries
                        sorted_rag = sorted(rag_results, key=lambda r: r['similarity'], reverse=True)
                        merged_segments = self.merge_adjacent_segments(sorted_rag, max_gap=3.0)
                        expanded_segments = self.expand_segments_for_completeness(merged_segments, max_expansion=60)
                        context_lines = [f"[{r['start']:.2f} - {r['end']:.2f}] {r['text']}" for r in expanded_segments]
                        rag_context = "\n".join(context_lines)
                else:
                    # Use segments as before for 4+ videos
                    if rag_results:
                        # For acronym queries, prioritize segments that contain the acronym
                        query_upper = query.upper()
                        detected_acronyms = re.findall(r'\b[A-Z]{2,}\b', query_upper)
                        common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION'}
                        detected_acronyms = [a for a in detected_acronyms if a not in common_words]
                        
                        if detected_acronyms:
                            # Prioritize segments that contain the acronym
                            acronym_segments = []
                            other_segments = []
                            
                            for r in rag_results:
                                if any(acronym in r['text'].upper() for acronym in detected_acronyms):
                                    acronym_segments.append(r)
                                else:
                                    other_segments.append(r)
                            
                            # Sort acronym segments by similarity, then add other segments
                            sorted_acronym = sorted(acronym_segments, key=lambda r: r['similarity'], reverse=True)
                            sorted_other = sorted(other_segments, key=lambda r: r['similarity'], reverse=True)
                            
                            # Combine with acronym segments first
                            combined_segments = sorted_acronym + sorted_other
                            print(f"Prioritized {len(sorted_acronym)} acronym-containing segments out of {len(combined_segments)} total")
                        else:
                            combined_segments = sorted(rag_results, key=lambda r: r['similarity'], reverse=True)
                        
                        merged_segments = self.merge_adjacent_segments(combined_segments, max_gap=3.0)
                        expanded_segments = self.expand_segments_for_completeness(merged_segments, max_expansion=60)
                        context_lines = [f"[{r['start']:.2f} - {r['end']:.2f}] {r['text']}" for r in expanded_segments]
                        rag_context = "\n".join(context_lines)
                    else:
                        rag_context = ""
                
                print(f"RAG context for clipping: {rag_context[:200]}...")
                
                response_text = chain_clipping.invoke({"query": query, "transcript": rag_context})
                response = extract_content(response_text)
                print(f"LLM Response: {response}")  # Debug
                
                # Parse and validate clips
                proposed_clips = self.parse_llm_clip_response(response)
                
                print(f"Proposed clips: {proposed_clips}")
                
                # Limit the number of clips to avoid overwhelming responses
                max_clips = 2  # Limit to 2 clips maximum
                if len(proposed_clips) > max_clips:
                    print(f"Too many clips ({len(proposed_clips)}), limiting to {max_clips} most relevant")
                    # Keep the first 2 clips (usually the most relevant ones)
                    proposed_clips = proposed_clips[:max_clips]
                
                # Deduplicate overlapping clips
                if len(proposed_clips) > 1:
                    print("Deduplicating overlapping clips...")
                    proposed_clips = self.deduplicate_clips(proposed_clips, min_overlap=0.3, query=query)
                    print(f"After deduplication: {len(proposed_clips)} clips")
                
                # Assign clips to their correct videos based on acronym content
                if any(keyword in query.lower() for keyword in ['acronym', 'meaning', 'what is', 'stands for']):
                    print("Assigning clips to correct videos...")
                    proposed_clips = self.assign_clips_to_correct_videos(proposed_clips, query, rag_results)
                
                # Validate that all requested acronyms are covered
                if self.is_definition_query(query):
                    all_covered = self.validate_all_acronyms_covered(query, proposed_clips, rag_results)
                    if not all_covered:
                        print("Warning: Not all requested acronyms are covered in the clips")
                
                video_clips = []
                relevant_video_ids = list({seg['video_id'] for seg in rag_results}) if rag_results else []
                available_videos = []
                for vid in relevant_video_ids:
                    candidate_path = os.path.join("Max Life Videos", f"{vid}.mp4")
                    if os.path.exists(candidate_path):
                        available_videos.append((vid, candidate_path, self.get_video_duration(candidate_path)))
                print(f"Available videos: {available_videos}")
                
                # Process each proposed clip
                for clip in proposed_clips:
                    start = clip['start']
                    end = clip['end']
                    print(f"Processing clip: {start}-{end}")
                    
                    video_file = None
                    matched_video_id = None
                    
                    # Use the assigned video_id if available
                    if 'video_id' in clip:
                        matched_video_id = clip['video_id']
                        print(f"Using assigned video_id: {matched_video_id}")
                    
                    # For acronym queries, try to match based on the acronym segments first
                    if not matched_video_id and any(keyword in query.lower() for keyword in ['acronym', 'meaning', 'what is', 'stands for']):
                        # Look for acronym segments that match this clip time range
                        for seg in rag_results:
                            if (start < seg['end'] and end > seg['start']):
                                matched_video_id = seg['video_id']
                                print(f"Matched clip to video {matched_video_id} based on acronym segment overlap")
                                break
                        
                        # If no match found, try to find video based on acronym content
                        if not matched_video_id:
                            query_acronyms = re.findall(r'\b[A-Z]{2,}\b', query.upper())
                            print(f"Trying to match clip to video based on acronyms: {query_acronyms}")
                            
                            # For each video, check if it contains the acronyms and if the clip time range makes sense
                            for video_id in video_ids:
                                full_transcript = self.load_full_transcript(video_id)
                                if full_transcript:
                                    transcript_upper = full_transcript.upper()
                                    # Check if this video contains any of the requested acronyms
                                    video_contains_acronyms = []
                                    for acronym in query_acronyms:
                                        if acronym in transcript_upper:
                                            video_contains_acronyms.append(acronym)
                                    
                                    if video_contains_acronyms:
                                        # Check if this clip time range is reasonable for this video
                                        video_duration = None
                                        for vid, path, dur in available_videos:
                                            if vid == video_id:
                                                video_duration = dur
                                                break
                                        
                                        if video_duration and start < video_duration:
                                            # Additional check: see if this video actually has content around this time
                                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                                            if video_segments:
                                                # Check if any segment from this video overlaps with the clip
                                                has_overlap = any(
                                                    (start < seg['end'] and end > seg['start']) 
                                                    for seg in video_segments
                                                )
                                                if has_overlap:
                                                    matched_video_id = video_id
                                                    print(f"Matched clip to video {matched_video_id} based on acronyms {video_contains_acronyms} and segment overlap")
                                                    break
                    
                    # If still no match, use the original logic
                    if not matched_video_id:
                        # Find the video that has the most relevant content for this clip time range
                        best_video = None
                        best_score = 0
                        best_overlap = 0
                        
                        for video_id in video_ids:
                            # Get all segments from this video
                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                            if not video_segments:
                                continue
                            
                            # Calculate overlap and relevance for this video
                            total_overlap = 0
                            total_relevance = 0
                            relevant_segments = 0
                            
                            for seg in video_segments:
                                # Calculate overlap between clip and segment
                                overlap_start = max(start, seg['start'])
                                overlap_end = min(end, seg['end'])
                                overlap_duration = max(0, overlap_end - overlap_start)
                                
                                if overlap_duration > 0:
                                    total_overlap += overlap_duration
                                    # Use similarity as relevance score (lower is better)
                                    relevance_score = 1 - seg['similarity']
                                    total_relevance += relevance_score * overlap_duration
                                    relevant_segments += 1
                            
                            if total_overlap > 0:
                                # Calculate average relevance weighted by overlap
                                avg_relevance = total_relevance / total_overlap
                                
                                # Score based on both overlap and relevance
                                # Prioritize videos with more overlap and better relevance
                                score = total_overlap * avg_relevance
                                
                                print(f"Video {video_id}: overlap={total_overlap:.2f}s, relevance={avg_relevance:.3f}, score={score:.3f}")
                                
                                if score > best_score:
                                    best_score = score
                                    best_video = video_id
                                    best_overlap = total_overlap
                        
                        if best_video:
                            matched_video_id = best_video
                            print(f"Matched clip to video {matched_video_id} based on content relevance (overlap: {best_overlap:.2f}s, score: {best_score:.3f})")
                        else:
                            # Fallback: find any video with segments in this time range
                            for seg in rag_results:
                                if (start < seg['end'] and end > seg['start']):
                                    matched_video_id = seg['video_id']
                                    print(f"Fallback: matched clip to video {matched_video_id} based on time overlap")
                                    break
                
                    # Fallback to first available video if still no match
                    if not matched_video_id and video_ids:
                        matched_video_id = video_ids[0]
                        print(f"Fallback: using first available video {matched_video_id}")
                    
                    # Try to get the video file
                    if matched_video_id:
                        candidate_path = os.path.join("Max Life Videos", f"{matched_video_id}.mp4")
                        if os.path.exists(candidate_path):
                            video_file = candidate_path
                    
                    # Fallback to any available video
                    if not video_file and available_videos:
                        video_file = available_videos[0][1]
                        print(f"Fallback: using first available video file")
                    
                    # Fallback to uploaded video
                    if not video_file and self.video_path and os.path.exists(self.video_path):
                        video_file = self.video_path
                        print(f"Fallback: using uploaded video")
                    
                    # 7. Validate and improve the clip
                    if video_file:
                        duration = self.get_video_duration(video_file)
                        print(f"Using video: {video_file} (duration: {duration}) for clip {start}-{end}")
                    
                    if duration is not None:
                        # Determine if this is an acronym query
                        is_acronym_query = any(keyword in query.lower() for keyword in ['acronym', 'meaning', 'what is', 'stands for']) and any(acronym in query.upper() for acronym in ['NOPP', 'ABCD', 'SMART', 'SWOT', 'ABCDEF'])
                        
                        # Add video_id to clip for natural ending detection
                        clip_with_video = clip.copy()
                        clip_with_video['video_id'] = matched_video_id
                        
                        # Validate and improve the clip
                        improved_clips = self.validate_and_improve_clips([clip_with_video], duration, is_acronym_query=is_acronym_query)
                        
                        for improved_clip in improved_clips:
                            print(f"Improved clip: {improved_clip['start']:.2f}-{improved_clip['end']:.2f} (duration: {improved_clip['duration']:.2f}s)")
                            msg, clip_path = self.clip_video(video_file, improved_clip['start'], improved_clip['end'])
                            if clip_path:
                                video_clips.append(clip_path)
                    else:
                        self.display_message("system", f"❌ Could not determine video duration for {video_file}")
                else:
                    self.display_message("system", "❌ No valid video found for this clip range.")
                
                video_clips = [p for p in video_clips if p and os.path.exists(p)]
                if not video_clips:
                    self.display_message("system", "⚠ No valid clips generated for concatenation.")
                else:
                    # Validate video compatibility before concatenation
                    is_compatible, compatibility_msg = self.validate_video_compatibility(video_clips)
                    if not is_compatible:
                        print(f"Warning: {compatibility_msg}")
                        self.display_message("system", f"⚠ Video compatibility warning: {compatibility_msg}")
                        # Continue anyway since we're using re-encoding
                    
                    output_filepath = os.path.join(tempfile.gettempdir(), "final_output.mp4")
                    msg, out = concatenate_videos(video_clips, output_filepath)
                    print(f"Concatenation result: {msg}, Output: {out}")  # Debug
                    self.display_message("Vivi", msg)
            else:
                context, rag_results = self.build_llm_context(user_input, for_clipping=False)
                
                # Debug: Show what context is being used for normal queries
                print(f"=== NORMAL QUERY DEBUG ===")
                print(f"Query: {user_input}")
                print(f"Context length: {len(context)}")
                print(f"RAG results count: {len(rag_results) if rag_results else 0}")
                
                # Use the debug helper
                self.debug_video_usage(rag_results, "normal")
                
                print(f"Context preview: {context[:300]}...")
                
                response_text = chain_general.invoke({
                    "context": self.context,
                    "question": user_input,
                    "transcript": context
                })
                response = extract_content(response_text)
                self.typewriter_effect("Vivi", response)
                self.context += f"\nUser: {user_input}\nAI: {response}\n"
            self.user_entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            self.user_entry.focus()

        threading.Thread(target=run_bot).start()

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
        if not self.video_path:
            return
        self.display_message("system", f"📁 Selected video: {os.path.basename(self.video_path)}")
        def process_video():
            self.full_transcript_text, self.transcript_segments = self.transcribe_video(self.video_path)
            print(f"Transcript segments: {self.transcript_segments}")  # Debug
            self.display_message("system", "✅ Transcription completed!")
            #self.progress_var.set(0)
        threading.Thread(target=process_video).start()

    def play_input_video(self):
        if not self.video_path:
            self.display_message("system", "⚠️ No video uploaded yet. Upload a video first.")
            return
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
        try:
            self.audio_process = subprocess.Popen(["ffplay", "-autoexit", "-loglevel", "quiet", self.video_path])
        except Exception as e:
            self.display_message("system", f"❌ Failed to play input video: {e}")

    def play_final_video(self):
        # Check temp directory first, then fallback to original video directory
        final_output = os.path.join(tempfile.gettempdir(), "final_output.mp4")
        if not os.path.exists(final_output) and self.video_path:
            final_output = os.path.join(os.path.dirname(self.video_path), "final_output.mp4")
        
        if not os.path.exists(final_output):
            self.display_message("system", "⚠️ No final video available. Generate a clipped video first.")
            return
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
        try:
            self.audio_process = subprocess.Popen(["ffplay", "-autoexit", "-loglevel", "quiet", final_output])
        except Exception as e:
            self.display_message("system", f"❌ Failed to play final video: {e}")

    def get_video_duration(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = None
            cap.release()
            return duration
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return None

    def get_video_properties(self, video_path):
        """Get video properties like codec, frame rate, resolution, etc."""
        try:
            command = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            properties = {
                'duration': None,
                'fps': None,
                'width': None,
                'height': None,
                'video_codec': None,
                'audio_codec': None,
                'audio_sample_rate': None,
                'audio_channels': None
            }
            
            if 'format' in data and 'duration' in data['format']:
                properties['duration'] = float(data['format']['duration'])
            
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video':
                    properties['fps'] = eval(stream.get('r_frame_rate', '30/1'))
                    properties['width'] = int(stream.get('width', 0))
                    properties['height'] = int(stream.get('height', 0))
                    properties['video_codec'] = stream.get('codec_name', 'unknown')
                elif stream['codec_type'] == 'audio':
                    properties['audio_codec'] = stream.get('codec_name', 'unknown')
                    properties['audio_sample_rate'] = int(stream.get('sample_rate', 0))
                    properties['audio_channels'] = int(stream.get('channels', 0))
            
            return properties
        except Exception as e:
            print(f"Error getting video properties for {video_path}: {e}")
            return None

    def validate_video_compatibility(self, video_paths):
        """Validate that all videos have compatible properties for concatenation."""
        if not video_paths:
            return True, "No videos to validate"
        
        properties_list = []
        for path in video_paths:
            props = self.get_video_properties(path)
            if props:
                properties_list.append((path, props))
            else:
                return False, f"Could not get properties for {path}"
        
        if len(properties_list) < 2:
            return True, "Only one video, no compatibility issues"
        
        # Check for major compatibility issues
        issues = []
        base_props = properties_list[0][1]
        
        for path, props in properties_list[1:]:
            # Check for major differences that could cause issues
            if props['video_codec'] != base_props['video_codec']:
                issues.append(f"Different video codecs: {base_props['video_codec']} vs {props['video_codec']} in {path}")
            
            if props['audio_codec'] != base_props['audio_codec']:
                issues.append(f"Different audio codecs: {base_props['audio_codec']} vs {props['audio_codec']} in {path}")
            
            if abs(props['fps'] - base_props['fps']) > 1:
                issues.append(f"Significant FPS difference: {base_props['fps']} vs {props['fps']} in {path}")
            
            if abs(props['audio_sample_rate'] - base_props['audio_sample_rate']) > 1000:
                issues.append(f"Different audio sample rates: {base_props['audio_sample_rate']} vs {props['audio_sample_rate']} in {path}")
        
        if issues:
            return False, f"Compatibility issues found: {'; '.join(issues)}"
        
        return True, "All videos are compatible"

    def deduplicate_clips(self, clips, min_overlap=0.3, query=""):
        """Remove overlapping clips and keep only the most relevant ones."""
        if not clips:
            return clips
        
        # Add duration to clips if not present
        for clip in clips:
            if 'duration' not in clip:
                clip['duration'] = clip['end'] - clip['start']
        
        # Sort clips by duration (longer clips first) and then by start time
        sorted_clips = sorted(clips, key=lambda x: (x['duration'], -x['start']), reverse=True)
        
        non_overlapping = []
        
        for clip in sorted_clips:
            is_duplicate = False
            
            for existing in non_overlapping:
                # Calculate overlap
                overlap_start = max(clip['start'], existing['start'])
                overlap_end = min(clip['end'], existing['end'])
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    clip_duration = clip['end'] - clip['start']
                    existing_duration = existing['end'] - existing['start']
                    
                    # Calculate overlap percentage
                    overlap_ratio = overlap_duration / min(clip_duration, existing_duration)
                    
                    # For acronym queries, be more careful about removing overlapping clips
                    # as they might cover different acronyms
                    if query and any(keyword in query.lower() for keyword in ['acronym', 'meaning', 'what is', 'stands for']):
                        # Only remove if overlap is very high (>80%) and clips are from same video
                        if overlap_ratio > 0.8 and clip.get('video_id') == existing.get('video_id'):
                            is_duplicate = True
                            print(f"Removing duplicate clip: {clip['start']:.2f}-{clip['end']:.2f} (overlaps {overlap_ratio:.2f} with {existing['start']:.2f}-{existing['end']:.2f})")
                            break
                    else:
                        # Normal deduplication for non-acronym queries
                        if overlap_ratio > min_overlap:
                            is_duplicate = True
                            print(f"Removing duplicate clip: {clip['start']:.2f}-{clip['end']:.2f} (overlaps {overlap_ratio:.2f} with {existing['start']:.2f}-{existing['end']:.2f})")
                            break
            
            if not is_duplicate:
                non_overlapping.append(clip)
                print(f"Keeping clip: {clip['start']:.2f}-{clip['end']:.2f} (duration: {clip['duration']:.2f}s)")
        
        return non_overlapping

    def validate_acronym_completeness(self, text, acronyms):
        """Validate that an acronym explanation is complete."""
        if not acronyms:
            return True
        
        text_upper = text.upper()
        completeness_score = 0
        
        for acronym in acronyms:
            # Check if the acronym is mentioned
            if acronym in text_upper:
                completeness_score += 1
                
                # Check if individual letters are explained
                for letter in acronym:
                    # Look for patterns that indicate letter explanation
                    patterns = [
                        f"{letter} is", f"{letter} stands for", f"{letter} means",
                        f"{letter} represents", f"{letter} refers to", f"{letter} indicates"
                    ]
                    if any(pattern in text_upper for pattern in patterns):
                        completeness_score += 0.5
        
        # Calculate completeness percentage
        max_score = len(acronyms) * 1.5  # Each acronym can contribute 1.5 points
        completeness_percentage = completeness_score / max_score if max_score > 0 else 0
        
        print(f"Acronym completeness: {completeness_percentage:.2f} ({completeness_score}/{max_score})")
        return completeness_percentage >= 0.6  # At least 60% complete

    def parse_llm_clip_response(self, response):
        """Parse LLM response to extract clip ranges more robustly."""
        proposed_clips = []
        
        # Split response into lines and look for clip ranges
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for different patterns of clip ranges
            patterns = [
                r"- Range:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"Range:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"start[:\s]*(\d+\.?\d*)[\s-]+end[:\s]*(\d+\.?\d*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        
                        # Basic validation
                        if start >= 0 and end > start and end - start <= 300:  # Max 5 minutes
                            proposed_clips.append({
                                'start': start, 
                                'end': end, 
                                'duration': end - start
                            })
                            print(f"Parsed clip: {start:.2f}-{end:.2f}")
                            break
                    except (ValueError, IndexError):
                        continue
        
        return proposed_clips

    def is_definition_query(self, query):
        """Detect if a query is asking for a definition or explanation."""
        query_lower = query.lower()
        definition_keywords = [
            'what is', 'what does', 'meaning of', 'definition of', 'explain', 'describe',
            'tell me about', 'how does', 'what are', 'define', 'explanation'
        ]
        return any(keyword in query_lower for keyword in definition_keywords)

    def assign_clips_to_correct_videos(self, clips, query, rag_results):
        """Assign clips to their correct videos based on acronym content."""
        if not clips or not any(keyword in query.lower() for keyword in ['acronym', 'meaning', 'what is', 'stands for']):
            return clips
        
        # Extract only actual acronyms from query (not common words)
        query_upper = query.upper()
        # Look for specific known acronyms first
        known_acronyms = ['NOPP', 'STARULIP', 'ULIP', 'SMART', 'SWOT', 'ABCD', 'ABCDEF']
        detected_acronyms = [acronym for acronym in known_acronyms if acronym in query_upper]
        
        # If no known acronyms found, look for general acronym patterns (3+ letters)
        if not detected_acronyms:
            general_acronyms = re.findall(r'\b[A-Z]{3,}\b', query_upper)
            # Filter out common words that might be mistaken for acronyms
            common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY'}
            detected_acronyms = [acronym for acronym in general_acronyms if acronym not in common_words]
        
        print(f"Assigning clips to videos based on actual acronyms: {detected_acronyms}")
        
        # If we have multiple acronyms, we need to assign clips to different videos
        if len(detected_acronyms) > 1:
            print(f"Multiple acronyms detected: {detected_acronyms}")
            # Create a mapping of acronyms to videos
            acronym_video_map = {}
            
            for acronym in detected_acronyms:
                best_video = None
                best_score = 0
                
                for video_id in list({r['video_id'] for r in rag_results}):
                    full_transcript = self.load_full_transcript(video_id)
                    if full_transcript:
                        transcript_upper = full_transcript.upper()
                        
                        # Check if this video contains this specific acronym
                        if acronym in transcript_upper:
                            # Calculate score based on how prominently this acronym appears
                            acronym_count = transcript_upper.count(acronym)
                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                            segment_score = sum(1 - s['similarity'] for s in video_segments) if video_segments else 0
                            
                            total_score = acronym_count * 10 + segment_score
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_video = video_id
                
                if best_video:
                    acronym_video_map[acronym] = best_video
                    print(f"Mapped {acronym} to {best_video} (score: {best_score:.2f})")
            
            # Now assign clips based on which acronym they best match
            assigned_clips = []
            for clip in clips:
                start, end = clip['start'], clip['end']
                best_acronym = None
                best_match_score = 0
                
                # Find which acronym this clip best represents by checking content overlap
                for acronym, video_id in acronym_video_map.items():
                    # Check if this clip overlaps with segments from this video
                    video_segments = [s for s in rag_results if s['video_id'] == video_id]
                    overlap_score = 0
                    
                    for seg in video_segments:
                        if (start < seg['end'] and end > seg['start']):
                            overlap_score += 1 - seg['similarity']
                    
                    # Additional check: see if the clip time range makes sense for this video
                    full_transcript = self.load_full_transcript(video_id)
                    if full_transcript:
                        # Check if this video actually contains the acronym around this time
                        transcript_upper = full_transcript.upper()
                        if acronym in transcript_upper:
                            # Give bonus points for acronym presence
                            overlap_score += 5
                    
                    if overlap_score > best_match_score:
                        best_match_score = overlap_score
                        best_acronym = acronym
                
                if best_acronym:
                    clip['video_id'] = acronym_video_map[best_acronym]
                    clip['target_acronym'] = best_acronym
                    print(f"Assigned clip {start:.2f}-{end:.2f} to video {clip['video_id']} for acronym {best_acronym} (score: {best_match_score:.2f})")
                else:
                    # Fallback: assign to the first available video
                    if acronym_video_map:
                        first_acronym = list(acronym_video_map.keys())[0]
                        clip['video_id'] = acronym_video_map[first_acronym]
                        clip['target_acronym'] = first_acronym
                        print(f"Fallback: assigned clip {start:.2f}-{end:.2f} to video {clip['video_id']}")
                
                assigned_clips.append(clip)
            
            return assigned_clips
        
        else:
            # Single acronym case - use the original logic
            assigned_clips = []
            
            for clip in clips:
                start, end = clip['start'], clip['end']
                best_video = None
                best_score = 0
                
                # Check each video to see which one contains the relevant acronyms
                for video_id in list({r['video_id'] for r in rag_results}):
                    full_transcript = self.load_full_transcript(video_id)
                    if full_transcript:
                        transcript_upper = full_transcript.upper()
                        
                        # Check if this video contains any of the requested acronyms
                        video_acronyms = [acronym for acronym in detected_acronyms if acronym in transcript_upper]
                        
                        if video_acronyms:
                            # Check if this video has segments that overlap with the clip
                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                            overlap_score = 0
                            
                            for seg in video_segments:
                                if (start < seg['end'] and end > seg['start']):
                                    overlap_score += 1 - seg['similarity']  # Lower similarity is better
                            
                            # Additional score for specific acronym matches
                            # Check if this video contains the main acronyms (NOPP, StarULIP, etc.)
                            main_acronyms = ['NOPP', 'STARULIP', 'ULIP', 'SMART', 'SWOT', 'ABCD']
                            for main_acronym in main_acronyms:
                                if main_acronym in transcript_upper:
                                    # Give bonus points for main acronym matches
                                    overlap_score += 10
                                    print(f"Found main acronym {main_acronym} in {video_id}, adding bonus score")
                            
                            if overlap_score > best_score:
                                best_score = overlap_score
                                best_video = video_id
                
                if best_video:
                    clip['video_id'] = best_video
                    print(f"Assigned clip {start:.2f}-{end:.2f} to video {best_video} (score: {best_score:.2f})")
                else:
                    # Fallback: use the first video that contains any acronym
                    for video_id in list({r['video_id'] for r in rag_results}):
                        full_transcript = self.load_full_transcript(video_id)
                        if full_transcript and any(acronym in full_transcript.upper() for acronym in detected_acronyms):
                            clip['video_id'] = video_id
                            print(f"Fallback: assigned clip {start:.2f}-{end:.2f} to video {video_id}")
                            break
                
                assigned_clips.append(clip)
            
            return assigned_clips

    def validate_all_acronyms_covered(self, query, clips, rag_results):
        """Validate that all acronyms mentioned in the query are covered by the clips."""
        # Extract only actual acronyms from query (not common words)
        query_upper = query.upper()
        # Look for specific known acronyms first
        known_acronyms = ['NOPP', 'STARULIP', 'ULIP', 'SMART', 'SWOT', 'ABCD', 'ABCDEF']
        requested_acronyms = [acronym for acronym in known_acronyms if acronym in query_upper]
        
        # If no known acronyms found, look for general acronym patterns (3+ letters)
        if not requested_acronyms:
            general_acronyms = re.findall(r'\b[A-Z]{3,}\b', query_upper)
            # Filter out common words that might be mistaken for acronyms
            common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY'}
            requested_acronyms = [acronym for acronym in general_acronyms if acronym not in common_words]
        
        if not requested_acronyms:
            return True
        
        print(f"Validating coverage of requested acronyms: {requested_acronyms}")
        
        # Check if all requested acronyms are covered in the clips
        covered_acronyms = set()
        
        for clip in clips:
            # Find the video that contains this clip
            clip_video_id = None
            for seg in rag_results:
                if (clip['start'] < seg['end'] and clip['end'] > seg['start']):
                    clip_video_id = seg['video_id']
                    break
            
            if clip_video_id:
                # Load the transcript for this video and check for acronyms
                full_transcript = self.load_full_transcript(clip_video_id)
                if full_transcript:
                    transcript_upper = full_transcript.upper()
                    for acronym in requested_acronyms:
                        if acronym in transcript_upper:
                            covered_acronyms.add(acronym)
                            print(f"Found {acronym} in {clip_video_id}")
        
        # Also check if the clips actually contain the acronym content
        # by looking at the actual clip time ranges and video content
        for clip in clips:
            start, end = clip['start'], clip['end']
            
            # Find which video this clip should be from based on content
            for video_id in list({r['video_id'] for r in rag_results}):
                full_transcript = self.load_full_transcript(video_id)
                if full_transcript:
                    # Check if this video contains the acronyms and if the clip range makes sense
                    transcript_upper = full_transcript.upper()
                    for acronym in requested_acronyms:
                        if acronym in transcript_upper:
                            # Check if this video has content around the clip time
                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                            if video_segments:
                                # Check if any segment overlaps with the clip
                                has_overlap = any(
                                    (start < seg['end'] and end > seg['start']) 
                                    for seg in video_segments
                                )
                                if has_overlap:
                                    covered_acronyms.add(acronym)
                                    print(f"Confirmed {acronym} coverage in {video_id} for clip {start:.2f}-{end:.2f}")
        
        missing_acronyms = set(requested_acronyms) - covered_acronyms
        if missing_acronyms:
            print(f"Warning: Missing acronyms: {missing_acronyms}")
            return False
        
        print(f"All requested acronyms covered: {covered_acronyms}")
        return True

    def run(self):
        """Run the chatbot application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup: stop Google Drive sync
            if self.drive_sync:
                print("🛑 Stopping Google Drive sync...")
                self.drive_sync.stop_watching()
                print("✅ Google Drive sync stopped")

    def stop_google_drive_sync(self):
        """Stop Google Drive sync if running."""
        if self.drive_sync:
            self.drive_sync.stop_watching()
            print("✅ Google Drive sync stopped")

    def manual_sync(self):
        """Manually trigger Google Drive sync."""
        if not self.drive_sync:
            self.display_message("assistant", "❌ Google Drive sync is not available. Please check your credentials and try again.")
            return
        
        def sync_worker():
            try:
                # Update status
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="🔄 Syncing...",
                    text_color="#ffc107"
                ))
                
                # Perform sync
                sync_result = self.drive_sync.sync_folder()
                
                # Update status
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="✅ Google Drive sync active",
                    text_color="#28a745"
                ))
                
                # Show sync results
                message = f"✅ Sync completed!\n📥 Downloaded: {len(sync_result['downloaded'])} files\n🔄 Updated: {len(sync_result['updated'])} files"
                self.root.after(0, lambda: self.display_message("assistant", message))
                
            except Exception as e:
                error_msg = f"❌ Sync failed: {str(e)}"
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="❌ Sync failed",
                    text_color="#dc3545"
                ))
                self.root.after(0, lambda: self.display_message("assistant", error_msg))
        
        # Run sync in background thread
        sync_thread = threading.Thread(target=sync_worker, daemon=True)
        sync_thread.start()

if __name__ == "__main__":
    import sys
    
    # Check if command line arguments are provided for terminal testing
    if len(sys.argv) > 1:
        # Terminal mode for testing
        print("=== ClipQuery Terminal Mode ===")
        
        # Initialize the chatbot
        app = ViviChatbot()
        
        # Get the query from command line
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        
        # Test the RAG pipeline directly
        if app.rag:
            print("\nTesting RAG pipeline...")
            rag_results = app.rag.query_videos(query, n_results=20)
            filtered = [r for r in rag_results if r['similarity'] <= app.similarity_threshold]
            
            print(f"Raw RAG results: {len(rag_results)}")
            print(f"Filtered results: {len(filtered)}")
            
            # Show debug info
            app.debug_video_usage(filtered, "terminal")
            
            # Test context building
            context, final_rag_results = app.build_llm_context(query, for_clipping=False)
            print(f"\nContext length: {len(context)}")
            print(f"Context preview: {context[:500]}...")
            
            # Test clipping context if it's a clipping query
            if query.lower().startswith("clipping:"):
                clipping_context, clipping_rag = app.build_llm_context(query, for_clipping=True)
                print(f"\nClipping context length: {len(clipping_context)}")
                print(f"Clipping context preview: {clipping_context[:500]}...")
        else:
            print("RAG pipeline not available")
    else:
        # GUI mode
        app = ViviChatbot()
        app.run()