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
    try:
        if not video_paths:
            return "❌ Error: No video clips provided.", None
        temp_dir = tempfile.gettempdir()
        list_file_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in video_paths:
                normalized_path = path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        command = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file_path, "-c", "copy", output_filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        os.remove(list_file_path)
        return f"✅ Concatenated video saved to: {output_filepath}", output_filepath
    except subprocess.CalledProcessError as e:
        return f"❌ FFmpeg error: {e.stderr}", None
    except Exception as e:
        return f"❌ Exception during concatenation: {str(e)}", None

# ------------------ LLM Setup ------------------
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama3-70b-8192", temperature=1)

general_template = """
You are Vivi, an expert and friendly video assistant chatbot.
You are having an ongoing conversation with the user. You have access to a full transcript of a video. If the user's question is about the video, answer helpfully and refer to timestamps if relevant.
If the question is general and not related to the video, just respond helpfully like a normal assistant.
---
Conversation History:
{context}
---
Full Transcript of the Video:
{transcript}
---
User:
{question}
---
Vivi:
"""

clipping_template = """
You are an expert video analysis assistant. Given a user query and the transcript of a video with timestamps, identify all timestamp ranges where the video content is relevant to the query. Return the result as a list of timestamp ranges (start and end times in seconds, formatted as plain numbers with exactly two decimal places, e.g., 31.00, 45.00). Each range must be followed by a brief explanation of its relevance. Ensure end_time is greater than start_time and both are non-negative. If no relevant segments are found, return an empty list with a note explaining why.

Query: {query}
Transcript: {transcript}

Return the result in the following format, with each range and explanation on separate lines:
- Range: start_time - end_time
  Relevance: [Brief explanation of why this range is relevant to the query]

Example:
- Range: 10.00 - 20.00
  Relevance: The segment discusses the query topic in detail.
- Range: 55.60 - 62.72
  Relevance: The segment mentions related concepts.
"""

prompt_general = ChatPromptTemplate.from_template(general_template)
prompt_clipping = ChatPromptTemplate.from_template(clipping_template)
chain_general = prompt_general | llm
chain_clipping = prompt_clipping | llm

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

        # # Progress bar
        # self.progress_var = tk.DoubleVar()
        # self.progress_bar = tk.ttk.Progressbar(self.root, orient="horizontal", mode="determinate", variable=self.progress_var)
        # self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))

        # Initial welcome message
        self.display_message("system", "🎬 Welcome to Vivi!\nType 'exit' to quit.\nUse format `Clipping:<your query>` for semantic editing.")

        # show buffering
        self.buffering_label = ctk.CTkLabel(self.root, text="Analyzing...", font=("Arial", 16))
        self.buffering_label.pack(pady=10)
        
        self.buffering_label.pack_forget()

        # RAG pipeline integration
        try:
            self.rag = VideoRAG()
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize RAG pipeline: {e}")
            self.rag = None
        self.similarity_threshold = 0.7
        self.max_videos_for_full_transcript = 4
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

    def clip_video(self, start_time, end_time):
        try:
            if start_time >= end_time:
                self.display_message("system", f"⚠ Invalid clip range: {start_time} >= {end_time}")
                return "Invalid time range", None
            clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), clip_filename)
            result = subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time),
                "-i", self.video_path, "-c", "copy", output_path
            ], capture_output=True, text=True, check=True)
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

    def build_llm_context(self, query):
        """Use RAG to find relevant videos/segments and build the transcript context for the LLM."""
        if self.rag:
            rag_results = self.rag.query_videos(query, n_results=20)
            filtered = [r for r in rag_results if r['similarity'] <= self.similarity_threshold]
            if not filtered:
                return "", []
            from collections import defaultdict
            video_to_segments = defaultdict(list)
            for r in filtered:
                video_to_segments[r['video_id']].append(r)
            if len(video_to_segments) > self.max_videos_for_full_transcript:
                context_lines = []
                for vid, segs in video_to_segments.items():
                    ranges = [(s['start'], s['end']) for s in segs]
                    segments = self.load_transcript_segments(vid, relevant_ranges=ranges)
                    for seg in segments:
                        context_lines.append(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")
                return "\n".join(context_lines), filtered
            else:
                context_lines = []
                for vid in video_to_segments:
                    segments = self.load_transcript_segments(vid)
                    for seg in segments:
                        context_lines.append(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")
                return "\n".join(context_lines), filtered
        else:
            return "", []

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
            # Always use RAG to build context
            context, rag_results = self.build_llm_context(user_input)
            # Do NOT display RAG results to the user
            if user_input.lower().startswith("clipping:"):
                query = user_input[len("clipping:"):].strip()
                # Use RAG context for clipping
                response_text = chain_clipping.invoke({"query": query, "transcript": context})
                response = extract_content(response_text)
                print(f"LLM Response: {response}")  # Debug
                video_clips = []
                for line in response.strip().split("\n"):
                    if line.startswith("- Range: "):
                        parts = line[len("- Range: "):].split(" - ")
                        try:
                            start = float(parts[0])
                            end = float(parts[1])
                            msg, clip_path = self.clip_video(start, end)
                            if clip_path:
                                video_clips.append(clip_path)
                        except:
                            continue
                if not video_clips:
                    self.display_message("system", "⚠ No valid clips generated for concatenation.")
                else:
                    output_filepath = os.path.join(os.path.dirname(self.video_path), "final_output.mp4")
                    msg, out = concatenate_videos(video_clips, output_filepath)
                    print(f"Concatenation result: {msg}, Output: {out}")  # Debug
                    self.display_message("Vivi", msg)
            else:
                # Use RAG context for all LLM queries
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

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ViviChatbot()
    app.run()