import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import tempfile
import uuid
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

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

        ffmpeg_path = "C:\\ffmpeg\\ffmpeg.exe"  

        # Create unique list file in temp dir
        temp_dir = tempfile.gettempdir()
        list_file_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")

        # Write the list of files to concatenate
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in video_paths:
                f.write(f"file '{path.replace('\\', '/')}'\n")

        # Run ffmpeg concat
        command = [
            ffmpeg_path, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file_path,
            "-c", "copy", output_filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Cleanup list file
        os.remove(list_file_path)

        if result.returncode != 0:
            return f"❌ FFmpeg error:\n{result.stderr}", None

        return f"✅ Concatenated video saved to: {output_filepath}", output_filepath

    except Exception as e:
        return f"❌ Exception during concatenation: {str(e)}", None


#def concatenate_videos(video_paths, output_path):
 #   ffmpeg_path = 'C:\\ffmpeg'
  #  try:
   #     if not video_paths:
    #        return "Error: No video clips provided.", None

     #   list_file = "concat_list.txt"
      #  with open(list_file, "w") as f:
       #     for path in video_paths:
        #        f.write(f"file '{path}'\n")

#        command = [
 #           ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
  #          "-i", list_file, "-c", "copy", output_path
   #     ]

    #    result = subprocess.run(command, capture_output=True, text=True)
     #   os.remove(list_file)
      #  if result.returncode != 0:
       #     return f"Error concatenating: {result.stderr}", None
        #return f"✅ Concatenated video saved to {output_path}", output_path

#    except Exception as e:
 #       return f"Error concatenating videos: {str(e)}", None

# ------------------ LLM Setup ------------------
llm = OllamaLLM(model='gemma3')

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
You are an expert video analysis assistant. Given a user query and the transcript of a video with timestamps, identify all timestamp ranges where the video content is relevant to the query. Return the result as a list of timestamp ranges (start and end times in seconds) and a brief explanation of why each range is relevant.

Return timestamps as plain numbers with two decimal places (e.g., 31.00, 45.00) without brackets or other characters. Ensure the format is consistent. Ensure end_time is greater than start_time and both are non-negative.

Query: {query}
Transcript: {transcript}

Return the result in the following format:
- Range: start_time - end_time
  Relevance: [Brief explanation of why this range is relevant to the query]
"""

prompt_general = ChatPromptTemplate.from_template(general_template)
prompt_clipping = ChatPromptTemplate.from_template(clipping_template)
chain_general = prompt_general | llm
chain_clipping = prompt_clipping | llm

# ------------------ Vivi GUI Class ------------------

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ViviChatbot:
    def __init__(self):
        self.video_path = ""
        self.context = ""
        self.transcript_segments = []
        self.full_transcript_text = ""
        self.cap = None
        self.playing = False
        self.current_frame = 0
        self.seek_scale = None
        self.audio_process = None

        self.root = ctk.CTk()
        self.root.title("Vivi Video Chatbot")
        self.root.geometry("900x700")
        self.root.configure(fg_color="#FFE6E6")
        
        # Chat area as scrollable frame
        self.chat_frame = ctk.CTkScrollableFrame(self.root, fg_color="#FFE6E6", width=800, height=450)
        self.chat_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.user_entry = ctk.CTkEntry(self.root, font=("Arial", 12))
        self.user_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.user_entry.bind("<Return>", lambda e: self.send_message())

        self.send_btn = ctk.CTkButton(self.root, text="Send", font=("Arial", 12), command=self.send_message)
        self.send_btn.pack(pady=(0, 10))

        btn_frame = ctk.CTkFrame(self.root)
        btn_frame.pack()

        self.browse_btn = ctk.CTkButton(btn_frame, text="📂 Upload Video", command=self.browse_video)
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        self.preview_btn = ctk.CTkButton(btn_frame, text="▶️ Preview Video", command=self.play_video)
        self.preview_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.root)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.progress_bar.set(0)  # Set initial value (0.0 to 1.0)

        self.add_message("Welcome to the Vivi Video Chatbot! Type 'exit' to quit.\nFor getting your video clipped, enter your query in the format:\nVideo clipping:<query>", is_user=False)

    def add_message(self, message, is_user):
        # Add a colored message bubble to the chat_frame
        msg_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        msg_container.pack(fill="x", anchor="e" if is_user else "w", pady=4, padx=4)
        chat_font = 16
        if is_user:
            # User message: right-aligned, red
            msg_frame = ctk.CTkFrame(msg_container, fg_color="#FF4C4C", corner_radius=15)
            msg_frame.pack(side="right", anchor="e", padx=(80, 0))
            msg_label = ctk.CTkLabel(msg_frame, text=message, font=ctk.CTkFont(size=chat_font), wraplength=400, justify="left", text_color="white")
            msg_label.pack(padx=15, pady=10)
            avatar = ctk.CTkLabel(msg_container, text="👤", font=ctk.CTkFont(size=20))
            avatar.pack(side="right", padx=(10, 0), anchor="ne")
        else:
            # Bot message: left-aligned, blue
            avatar = ctk.CTkLabel(msg_container, text="🤖", font=ctk.CTkFont(size=20))
            avatar.pack(side="left", padx=(0, 10), anchor="nw")
            msg_frame = ctk.CTkFrame(msg_container, fg_color="#1E90FF", corner_radius=15)
            msg_frame.pack(side="left", anchor="w", padx=(0, 80))
            msg_label = ctk.CTkLabel(msg_frame, text=message, font=ctk.CTkFont(size=chat_font), wraplength=400, justify="left", text_color="white")
            msg_label.pack(padx=15, pady=10)
        self.root.update_idletasks()
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def clip_video(self, start_time, end_time):
        try:
            # Validate time range
            if start_time >= end_time:
                self.add_message(f"\n⚠ Invalid clip range: start ({start_time}) >= end ({end_time})\n", is_user=False)
                return " Invalid time range", None
            
            # Use a unique filename to avoid collisions
            clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), clip_filename)

            if os.path.exists(output_path):
                os.remove(output_path)

            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", self.video_path,
                "-c", "copy",
                output_path
            ], check=True)

            self.add_message(f"🎬 Video clip saved to {output_path}", is_user=False)
            return f"🎬 Video clip saved to {output_path}", output_path

        except subprocess.CalledProcessError as e:
            self.add_message(f"\n❌ FFmpeg failed: {e}\n", is_user=False)
        except PermissionError as e:
            self.add_message(f"\n❌ Permission denied: {e}\n", is_user=False)
        except Exception as e:
            self.add_message(f"\n❌ Error clipping video: {e}\n", is_user=False)
    
    def transcribe_video(self, video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        dir_ = os.path.dirname(video_path)
        txt_path = os.path.join(dir_, f"{base}.txt")
        srt_path = os.path.join(dir_, f"{base}.srt")

        if os.path.exists(srt_path):
            self.add_message("✅ Transcript and subtitles found.\n", is_user=False)
            with open(srt_path, "r", encoding="utf-8") as f:
                blocks = f.read().strip().split("\n\n")
            segments = []
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 3:
                    times = lines[1].split(" --> ")
                    start = _parse_srt_time(times[0])
                    end = _parse_srt_time(times[1])
                    text = " ".join(lines[2:])
                    segments.append({"start": start, "end": end, "text": text})
            return "\n".join(s['text'] for s in segments), segments

        self.add_message("🔍 Running Whisper transcription...\n", is_user=False)
        model = whisper.load_model("medium")
        result = model.transcribe(video_path, verbose=True)
        segments = []
        for seg in result['segments']:
            segments.append({"start": seg['start'], "end": seg['end'], "text": seg['text']})

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")

        return "\n".join(s['text'] for s in segments), segments

    def send_message(self):
        user_input = self.user_entry.get()
        if user_input.strip().lower() == "exit":
            self.root.destroy()
            return
        self.add_message(f"You: {user_input}", is_user=True)
        self.user_entry.delete(0, tk.END)
        self.user_entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        def run_bot():
            if user_input.lower().startswith("video clipping:"):
                query = user_input[len("video clipping:"):].strip()
                response = chain_clipping.invoke({"query": query, "transcript": self.transcript_segments})
                self.add_message(f"Vivi:", is_user=False)
                video_clips = []
                for line in response.strip().split("\n"):
                    if line.startswith("- Range: "):
                        parts = line[len("- Range: "):].split(" - ")
                        if len(parts) == 2:
                            try:
                                start = float(parts[0])
                                end = float(parts[1])
                            except ValueError:
                                try:
                                    start = _parse_srt_time(parts[0])
                                    end = _parse_srt_time(parts[1])
                                except Exception as e:
                                    self.add_message(f"⚠️ Failed to parse timestamps: {parts}", is_user=False)
                                    continue
                            msg, clip_path = self.clip_video(start, end)
                            self.add_message(msg, is_user=False)
                            if clip_path:
                                video_clips.append(clip_path)
                if video_clips:
                    msg, out = concatenate_videos(video_clips, output_filepath=os.path.join(os.path.dirname(self.video_path), "final_output.mp4"))
                    self.add_message(msg, is_user=False)
            else:
                response = chain_general.invoke({
                    "context": self.context,
                    "question": user_input,
                    "transcript": self.transcript_segments
                })
                self.add_message(response, is_user=False)
                self.context += f"\nUser: {user_input}\nAI: {response}\n"
            self.user_entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            self.user_entry.focus()

        threading.Thread(target=run_bot).start()

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
        if not self.video_path:
            return

        self.add_message(f"\n📁 Selected video: {os.path.basename(self.video_path)}\n", is_user=False)

        def process_video():
            self.full_transcript_text, self.transcript_segments = self.transcribe_video(self.video_path)
            self.add_message("✅ Transcription completed!\n\n", is_user=False)
            self.progress_var.set(0)

        threading.Thread(target=process_video).start()

    def play_video(self):
        if not self.video_path:
            self.add_message("\n⚠️ No video loaded yet. Upload a video first.\n", is_user=False)
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.add_message("\n❌ Failed to open video.\n", is_user=False)
            return

        self.cap = cap
        self.current_frame = 0
        self.playing = True

        # Kill previous audio if running
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()

        # Play audio using ffplay
        self.audio_process = subprocess.Popen([
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.video_path
        ])

        self.video_frame = ctk.CTkToplevel(self.root)
        self.video_frame.title("🎥 Video Player")

        self.canvas = tk.Canvas(self.video_frame, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.canvas.pack()

        control_frame = ctk.CTkFrame(self.video_frame)
        control_frame.pack()
        play_btn = ctk.CTkButton(control_frame, text="▶️", command=lambda: self._play_frames)
        play_btn.pack(side="left", padx=5)
        pause_btn = ctk.CTkButton(control_frame, text="⏸️ Pause", command=self.pause_video)
        pause_btn.pack(side="left", padx=5)

        self.seek_scale = ctk.CTkSlider(
            control_frame,
            from_=0,
            to=self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            orient="horizontal",
            length=400,
            command=self.seek_video
        )
        self.seek_scale.pack(side="left", padx=10)

        self.playing = True
        self._play_frames()

    def _play_frames(self):
        if not self.playing or self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            if self.audio_process and self.audio_process.poll() is None:
                self.audio_process.terminate()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.img = img
        self.canvas.create_image(0, 0, anchor="nw", image=img)

        self.current_frame += 1
        self.seek_scale.set(self.current_frame)

        delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
        self.canvas.after(delay, self._play_frames)

    def pause_video(self):
        self.playing = False
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()

    def resume_video(self):
        if not self.playing:
            self.playing = True
            self._play_frames()

    def seek_video(self, value):
        self.current_frame = int(float(value))
        if not self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.img = img
                self.canvas.create_image(0, 0, anchor="nw", image=img)

    def run(self):
        self.root.mainloop()

# Start the chatbot
if __name__ == "__main__":
    app = ViviChatbot()
    app.run()