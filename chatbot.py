#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from faster_whisper import WhisperModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import tempfile
import uuid
import time
import torch
import cv2
from PIL import Image, ImageTk


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
llm = OllamaLLM(model='gemma3:1B')

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

        self.root = tk.Tk()
        self.root.title("Vivi Video Chatbot")

        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=30, font=("Arial", 12))
        self.chat_display.pack(padx=10, pady=10)

        self.user_entry = tk.Entry(self.root, font=("Arial", 12))
        self.user_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.user_entry.bind("<Return>", lambda e: self.send_message())

        self.send_btn = tk.Button(self.root, text="Send", font=("Arial", 12), command=self.send_message)
        self.send_btn.pack(pady=(0, 10))

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        self.browse_btn = tk.Button(btn_frame, text="📂 Upload Video", command=self.browse_video)
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        self.preview_btn = tk.Button(btn_frame, text="▶️ Preview Video", command=self.play_video)
        self.preview_btn.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        self.chat_display.insert(tk.END, " Welcome to the Vivi Video Chatbot! Type 'exit' to quit.\n For getting your video clipped, enter your query in the format:\n Video clipping:<query>")
        self.chat_display.yview(tk.END)

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

    def clip_video(self, start_time, end_time):
        try:
            # Validate time range
            if start_time >= end_time:
                self.chat_display.insert(tk.END, f"\n⚠ Invalid clip range: start ({start_time}) >= end ({end_time})\n")
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

            self.chat_display.yview(tk.END)
            return f"🎬 Video clip saved to {output_path}", output_path

        except subprocess.CalledProcessError as e:
            self.chat_display.insert(tk.END, f"\n❌ FFmpeg failed: {e}\n")
        except PermissionError as e:
            self.chat_display.insert(tk.END, f"\n❌ Permission denied: {e}\n")
        except Exception as e:
            self.chat_display.insert(tk.END, f"\n❌ Error clipping video: {e}\n")

    def transcribe_video(self, video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        dir_ = os.path.dirname(video_path)
        txt_path = os.path.join(dir_, f"{base}.txt")
        srt_path = os.path.join(dir_, f"{base}.srt")

        if os.path.exists(txt_path) and os.path.exists(srt_path):
            self.chat_display.insert(tk.END, "✅ Transcript and subtitles found.\n")
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

        self.chat_display.insert(tk.END, "🔍 Running Whisper transcription...\n")
        model = WhisperModel("medium", device="cuda" if torch.cuda.is_available() else "cpu")
        segments_iter, info = model.transcribe(video_path, beam_size=5)
        segments = []
        for seg in segments_iter:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")

        return "\n".join(s['text'] for s in segments), segments

    def send_message(self):
        user_input = self.user_entry.get()
        if user_input.strip().lower() == "exit":
            self.root.destroy()
            return

        self.chat_display.insert(tk.END, f"You: {user_input}\n")
        self.user_entry.delete(0, tk.END)
        self.user_entry.config(state="disabled")
        self.send_btn.config(state="disabled")

        def run_bot():
            if user_input.lower().startswith("video clipping:"):
                query = user_input[len("video clipping:"):].strip()
                response = chain_clipping.invoke({"query": query, "transcript": self.transcript_segments})
                self.chat_display.insert(tk.END, f"\nVivi:")
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
                                    self.chat_display.insert(tk.END, f"\n⚠️ Failed to parse timestamps: {parts}")
                                    continue
                            msg, clip_path = self.clip_video(start, end)
                            self.chat_display.insert(tk.END, f"\n{msg}")
                            if clip_path:
                                video_clips.append(clip_path)
                if video_clips:
                    msg, out = concatenate_videos(video_clips, output_filepath=os.path.join(os.path.dirname(self.video_path), "final_output.mp4"))
                    self.chat_display.insert(tk.END, f"\n{msg}\n")
            else:
                response = chain_general.invoke({
                    "context": self.context,
                    "question": user_input,
                    "transcript": self.transcript_segments
                })
                self.chat_display.insert(tk.END, "Vivi: ")
                for word in response.split():
                    self.chat_display.insert(tk.END, word + " ")
                    self.chat_display.yview(tk.END)
                    time.sleep(0.04)
                self.chat_display.insert(tk.END, "\n\n")
                self.context += f"\nUser: {user_input}\nAI: {response}\n"
            self.user_entry.config(state="normal")
            self.send_btn.config(state="normal")
            self.user_entry.focus()

        threading.Thread(target=run_bot).start()

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
        if not self.video_path:
            return

        self.chat_display.insert(tk.END, f"\n📁 Selected video: {os.path.basename(self.video_path)}\n")

        def process_video():
            self.full_transcript_text, self.transcript_segments = self.transcribe_video(self.video_path)
            self.chat_display.insert(tk.END, "✅ Transcription completed!\n\n")
            self.progress_var.set(0)

        threading.Thread(target=process_video).start()

    def play_video(self):
        if not self.video_path:
            self.chat_display.insert(tk.END, "\n⚠️ No video loaded yet. Upload a video first.\n")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.chat_display.insert(tk.END, "\n❌ Failed to open video.\n")
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

        self.video_frame = tk.Toplevel(self.root)
        self.video_frame.title("🎥 Video Player")

        self.canvas = tk.Canvas(self.video_frame, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.canvas.pack()

        control_frame = tk.Frame(self.video_frame)
        control_frame.pack()
        play_btn = tk.Button(control_frame, text="▶️", command=lambda: self._play_frames)
        play_btn.pack(side="left", padx=5)
        pause_btn = tk.Button(control_frame, text="⏸️ Pause", command=self.pause_video)
        pause_btn.pack(side="left", padx=5)

        self.seek_scale = ttk.Scale(
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
    try:
        app = ViviChatbot()
        app.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")


# In[ ]:




