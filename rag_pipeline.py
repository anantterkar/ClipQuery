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
        # Check if this is an acronym query
        query_upper = query.upper()
        # Look for acronym patterns (3+ capital letters)
        acronym_pattern = re.findall(r'\b[A-Z]{3,}\b', query_upper)
        
        # Common words to exclude from acronym detection
        common_words = {'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION', 'MEANING', 'DEFINITION', 'STANDS', 'ABOUT'}
        
        # Filter out common words
        acronyms = [acronym for acronym in acronym_pattern if acronym not in common_words]
        
        # Additional check: only treat as acronym query if the query specifically asks for explanation/meaning
        is_explanation_query = any(keyword in query_upper for keyword in ['EXPLAIN', 'MEANING', 'DEFINITION', 'STANDS', 'WHAT IS', 'TELL ME ABOUT'])
        
        if acronyms and is_explanation_query:
            print(f"Acronym explanation query detected: {acronyms}")
            # For acronym queries, prioritize exact matches
            return self._search_acronym_segments(acronyms, n_results)
        else:
            # For regular queries, use semantic similarity
            print(f"Regular semantic query detected")
            return self._search_semantic_segments(query, n_results)
    
    def _search_acronym_segments(self, acronyms: List[str], n_results: int) -> List[Dict[str, Any]]:
        """Search for segments containing specific acronyms, prioritizing exact matches"""
        # Get all segments from the collection
        all_results = self.segment_collection.get()
        
        if not all_results["ids"]:
            return []
        
        # Score segments based on acronym presence
        scored_segments = []
        
        for i in range(len(all_results["ids"])):
            segment_text = all_results["documents"][i]
            segment_text_upper = segment_text.upper()
            
            # Check for exact acronym matches
            exact_matches = []
            for acronym in acronyms:
                if acronym in segment_text_upper:
                    exact_matches.append(acronym)
            
            # Calculate score based on acronym matches
            if exact_matches:
                # Higher score for more acronym matches and longer segments (more context)
                score = len(exact_matches) * 0.5 + (len(segment_text) / 1000) * 0.3
                
                # Bonus for segments that contain multiple requested acronyms
                if len(exact_matches) > 1:
                    score += 0.2
                
                # Bonus for segments that seem to be explaining the acronym (contain keywords)
                explanation_keywords = ['MEANS', 'STANDS', 'REFERS', 'DEFINED', 'EXPLAIN', 'DESCRIBE', 'ABOUT', 'CRITERIA', 'NEED', 'OPPORTUNITY', 'PHYSICALLY', 'PAYING', 'CAPACITY']
                keyword_bonus = sum(0.1 for keyword in explanation_keywords if keyword in segment_text_upper)
                score += keyword_bonus
                
                # Debug: show what we found
                print(f"Found acronym segment: {segment_text[:100]}... (acronyms: {exact_matches}, score: {score:.3f})")
                
                scored_segments.append({
                    "video_id": all_results["metadatas"][i]["video_id"],
                    "text": segment_text,
                    "start": all_results["metadatas"][i]["start"],
                    "end": all_results["metadatas"][i]["end"],
                    "timestamp": all_results["metadatas"][i]["timestamp"],
                    "similarity": 1.0 - score,  # Convert to distance (lower is better)
                    "exact_matches": exact_matches,
                    "score": score
                })
        
        # Sort by score (higher score = better match)
        scored_segments.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Found {len(scored_segments)} segments containing acronyms: {acronyms}")
        
        # If no acronym segments found, fall back to semantic search
        if not scored_segments:
            print(f"No segments found containing acronyms {acronyms}, falling back to semantic search")
            return self._search_semantic_segments(f"What is {' '.join(acronyms)}?", n_results)
        
        # Take top n_results
        return scored_segments[:n_results]
    
    def _search_semantic_segments(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search for segments using semantic similarity (original method)"""
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