from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import subprocess

# encoding query and extracting similar segments from transcript
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_transcript_segments(text_query, transcript_segments, top_k=5):
    query_embedding = embedder.encode(text_query, convert_to_tensor=True)
    texts = [seg["text"] for seg in transcript_segments]
    text_embeddings = embedder.encode(texts, convert_to_tensor=True, batch_size=32)

    similarities = util.cos_sim(query_embedding, text_embeddings)[0]
    ranked = sorted(zip(transcript_segments, similarities), key=lambda x: x[1], reverse=True)

    results = []
    for segment, score in ranked[:top_k]:
        results.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"],
            "similarity": float(score)
        })
    return results

def get_major_topics(full_transcript_text):
    try:
        response = chain_topic.invoke({"transcript": full_transcript_text})
        topics = []
        lines = response.split("\n")
        current_topic = None
        for line in lines:
            if line.startswith("- Topic:"):
                current_topic = {"description": line.replace("- Topic:", "").strip()}
            elif line.startswith("  Timestamps:") and current_topic:
                times = line.replace("  Timestamps:", "").strip().split(" - ")
                if len(times) == 2:
                    current_topic["start"] = float(times[0])
                    current_topic["end"] = float(times[1])
                    topics.append(current_topic)
                    current_topic = None
        return topics
    except Exception as e:
        return [{"description": f"Error extracting topics: {str(e)}", "start": 0, "end": 0}]

def find_most_similar_topic(query, topics):
    if not topics:
        return None, 0.0
    query_embedding = embedder.encode(query)
    topic_descriptions = [topic["description"] for topic in topics]
    topic_embeddings = embedder.encode(topic_descriptions)
    similarities = util.cos_sim(query_embedding, topic_embeddings)[0]
    max_idx = similarities.argmax()
    return topics[max_idx], float(similarities[max_idx])

llm = OllamaLLM(model='gemma3')

# template 
general_template = """
You are Vivi, an expert and friendly video assistant chatbot.

You are having an ongoing conversation with the user. You have access to a full transcript of a video. If the user’s question is about the video, answer helpfully and refer to timestamps if relevant.

If the question is general and not related to the video, just respond helpfully like a normal assistant. You can use your general knowledge to help the user even if it’s unrelated to the transcript.

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
# Template for topic extraction
topic_template = """
You are an expert video analysis assistant. Given the transcript of a video with timestamps, identify the major topics discussed and provide their corresponding timestamp ranges. Return the result as a list of topics, each with a brief description and its start and end timestamps.

Transcript:
{transcript}

Return the result in the following format:
- Topic: [Brief description]
  Timestamps: [start_time] - [end_time]
"""

prompt_general = ChatPromptTemplate.from_template(general_template)
prompt_topic = ChatPromptTemplate.from_template(topic_template)
chain_general = prompt_general | llm
chain_topic = prompt_topic | llm


def answer_query_with_ollama(transcript, query, context):
    inputs = {
        "transcript": transcript,
        "question": query,
        "context": context or ""
    }
    
    response = chain_general.invoke(inputs)
    
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    return str(response)

# FFmpeg-based video clipping function
def clip_video(video_path, start_time, end_time, output_path=None):
    """
    Clip a video from start_time to end_time using FFmpeg.
    
    Args:
        video_path (str): Path to the input video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_path (str): Path to save the clipped video. If None, generates a default name.
    
    Returns:
        str: Success message with output path or error message.
    """
    ffmpeg_path = '/Users/mohalsahai/Desktop/Video Clipping/Python Packages/ffmpeg'
    try:
        if not os.path.exists(video_path):
            return f"Error: Video file {video_path} does not exist."
        
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_clip_{int(start_time)}_{int(end_time)}.mp4"
        
        # Calculate duration
        duration = end_time - start_time
        
        # FFmpeg command: -ss for start, -t for duration, -c copy for stream copying
        command = [
            ffmpeg_path, "-y",  # Overwrite output if exists
            "-i", video_path,   # Input file
            "-ss", str(start_time),  # Start time
            "-t", str(duration),     # Duration
            "-c", "copy",       # Copy streams without re-encoding
            output_path         # Output file
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"Video clip saved to {output_path}"
    
    except subprocess.CalledProcessError as e:
        return f"Error clipping video: FFmpeg failed with {e.stderr}"
    except Exception as e:
        return f"Error clipping video: {str(e)}"