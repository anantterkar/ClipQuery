import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any
import json
from datetime import datetime
import re

class VectorStore:
    def __init__(self, persist_directory: str = "vector_store"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collections
        self.video_collection = self.client.get_or_create_collection(
            name="videos",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.segment_collection = self.client.get_or_create_collection(
            name="segments",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embeddings model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_transcripts()
        
    def load_transcripts(self):
        """Load and process transcript files from Max Life Videos folder"""
        transcripts_folder = "Max Life Videos"
        
        if not os.path.exists(transcripts_folder):
            print(f"Transcripts folder '{transcripts_folder}' not found!")
            return
            
        # Get all .txt files in the folder
        txt_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"No transcript files found in '{transcripts_folder}' folder!")
            return
            
        print(f"Found {len(txt_files)} transcript files")
        
        for txt_file in txt_files:
            video_name = txt_file.replace('.txt', '')
            file_path = os.path.join(transcripts_folder, txt_file)
            
            print(f"Processing {video_name}...")
            
            # Add video to collection if not exists
            if not self.video_collection.get(ids=[video_name])["ids"]:
                self.video_collection.add(
                    ids=[video_name],
                    documents=[video_name],
                    metadatas=[{
                        "name": video_name,
                        "processed_at": datetime.now().isoformat()
                    }]
                )
            
            # Process transcript file
            self.process_transcript_file(file_path, video_name)
            
        print("Transcript processing completed!")
        
    def process_transcript_file(self, file_path: str, video_name: str):
        """Process a single transcript file and extract segments"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into lines and process each line
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse timestamp and text using regex
                match = re.match(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(.+)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    text = match.group(3).strip()
                    
                    if text:  # Only process if there's actual text content
                        # Generate embedding for the text
                        embedding = self.embedding_model.encode(text)
                        
                        # Create segment ID
                        segment_id = f"{video_name}_{start_time}_{end_time}"
                        
                        # Format timestamp for display
                        timestamp = f"[{start_time:.2f} - {end_time:.2f}]"
                        
                        # Add segment to collection
                        self.segment_collection.add(
                            ids=[segment_id],
                            documents=[text],
                            embeddings=[embedding.tolist()],
                            metadatas=[{
                                "video_id": video_name,
                                "start": start_time,
                                "end": end_time,
                                "timestamp": timestamp
                            }]
                        )
                        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise
        
    def search_segments(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant video segments"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search segments
        results = self.segment_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "video_id": results["metadatas"][0][i]["video_id"],
                "text": results["documents"][0][i],
                "start": results["metadatas"][0][i]["start"],
                "end": results["metadatas"][0][i]["end"],
                "timestamp": results["metadatas"][0][i]["timestamp"],
                "similarity": results["distances"][0][i]
            })
            
        return formatted_results

class VideoRAG:
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store = VectorStore(vector_store_path)
        
    def query_videos(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the video database"""
        return self.vector_store.search_segments(query, n_results) 