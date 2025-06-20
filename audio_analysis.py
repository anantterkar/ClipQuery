import os
import numpy as np
import torch
# CLIP is not currently used, so we'll make it optional
try:
    import clip
except ImportError:
    pass
from PIL import Image
import cv2
import whisper
import re
import requests
from pytubefix import YouTube
from moviepy import VideoFileClip
import subprocess
import argparse
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util
import time
import shutil
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def download_video_from_youtube(url, output_path="input_video.mp4"):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
        stream.download(filename=output_path)
        return output_path
    except RuntimeError:
        raise FileNotFoundError("File does not exist.")

def download_video_from_url(url, output_path="input_video.mp4"):
    if "youtube.com" in url or "youtu.be" in url:
        return download_video_from_youtube(url, output_path)
    
    elif "drive.google.com" in url:
        try:
            file_id = url.split("/d/")[1].split("/")[0]
        except IndexError:
            raise ValueError("Google Drive link format incorrect.")
        d_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(d_url, stream=True)
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_path

    else: 
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise ValueError("Could not download video from direct link.")
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_path

def convert_to_mp4(input_path, output_path="input_video.mp4"):
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        clip.close()
    except Exception as e:
        raise RuntimeError(f"Video conversion failed: {e}")

def extract_audio(video_path, audio_path="audio.wav"):
    print(f"Extracting audio from: {video_path}")
    clip = VideoFileClip(video_path)
    if not clip.audio:
        raise ValueError("No audio stream found in video.")
    clip.audio.write_audiofile(audio_path)
    print(f"Audio saved to: {audio_path}")
    return audio_path

def extract_subtitles(video_path, subtitle_path="subtitles.srt"):
    print("Trying to extract subtitles...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", subtitle_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if "Stream mapping:" in result.stderr and os.path.exists(subtitle_path):
        print(f"Subtitles saved to: {subtitle_path}")
        return subtitle_path
    else:
        print("No subtitles found in the video.")
        return None

 
def transcribe_audio(audio_path, output_dir, query=None):
    model = whisper.load_model("medium")
    result = model.transcribe("audio.wav", verbose=True, task='transcribe', language='en', fp16=False)
    
    timestamped_transcript=""
    for segment in result["segments"]:
        timestamped_transcript+=f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
        
    # save the transcript file
    transcript_path = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.txt")
    with open(transcript_path, 'w', encoding='uft-8') as f:
        f.write(result["text"])
        
    # function to format time
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")
    # save the srt file too
    srt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(transcript_path))[0] + ".srt")
    segments = result["segments"]
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            start = seg['start']
            end = seg['end']
            text = seg['text']
            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text.strip()}\n\n")
            
    return srt_path
     
