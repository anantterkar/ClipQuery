import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import shutil

# Google Drive API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

# For file watching and transcription
import whisper
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import re
import pickle

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

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
        
    def authenticate(self):
        """Authenticate with Google Drive API."""
        creds = None
        
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception as e:
                print(f"⚠️ Error loading existing token: {e}")
                print("🗑️ Removing invalid token file...")
                os.remove('token.json')
                creds = None
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Error refreshing token: {e}")
                    print("🗑️ Removing invalid token file...")
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    creds = None
            
            if not creds:
                print("🔐 Starting Google Drive authentication...")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                    # Save the credentials for the next run
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                    print("✅ Authentication successful!")
                except Exception as e:
                    print(f"❌ Authentication failed: {e}")
                    raise
        
        return build('drive', 'v3', credentials=creds)
    
    def list_drive_folders(self, service):
        """List all folders in Google Drive."""
        try:
            results = service.files().list(
                q="mimeType='application/vnd.google-apps.folder'",
                pageSize=50,
                fields="nextPageToken, files(id, name)"
            ).execute()
            folders = results.get('files', [])
            
            if not folders:
                print('No folders found in Google Drive.')
                return []
            else:
                print('Available folders in Google Drive:')
                for i, folder in enumerate(folders):
                    print(f"{i+1}. {folder['name']} (ID: {folder['id']})")
                return folders
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def choose_folder(self, folders):
        """Let user choose a folder from the list."""
        while True:
            try:
                choice = input(f"\nEnter folder number (1-{len(folders)}) or 'create' to create a new 'Max Life Videos' folder: ").strip()
                
                if choice.lower() == 'create':
                    return self.create_max_life_videos_folder()
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(folders):
                    selected_folder = folders[choice_num - 1]
                    print(f"Selected folder: {selected_folder['name']}")
                    return selected_folder
                else:
                    print(f"Please enter a number between 1 and {len(folders)}")
            except ValueError:
                print("Please enter a valid number or 'create'")
    
    def create_max_life_videos_folder(self, service):
        """Create a new 'Max Life Videos' folder in Google Drive."""
        try:
            folder_metadata = {
                'name': 'Max Life Videos',
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=folder_metadata, fields='id, name').execute()
            print(f"Created new folder: {folder['name']} (ID: {folder['id']})")
            return folder
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
    
    def download_folder_contents(self, service, folder_id, folder_name):
        """Download all files from the specified Google Drive folder."""
        try:
            # Create local folder if it doesn't exist
            local_folder = folder_name
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            
            # List all files in the folder
            results = service.files().list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            files = results.get('files', [])
            
            if not files:
                print(f'No files found in folder "{folder_name}".')
                return
            
            print(f"Found {len(files)} files in folder '{folder_name}':")
            for file in files:
                print(f"  - {file['name']} ({file['mimeType']})")
            
            # Download files
            for file in files:
                file_path = os.path.join(local_folder, file['name'])
                
                # Skip if file already exists
                if os.path.exists(file_path):
                    print(f"  Skipping {file['name']} (already exists)")
                    continue
                
                try:
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        # Recursively download subfolder
                        print(f"  Downloading subfolder: {file['name']}")
                        self.download_folder_contents(service, file['id'], os.path.join(local_folder, file['name']))
                    else:
                        # Download file
                        print(f"  Downloading: {file['name']}")
                        request = service.files().get_media(fileId=file['id'])
                        with open(file_path, 'wb') as f:
                            f.write(request.execute())
                except Exception as e:
                    print(f"  Error downloading {file['name']}: {e}")
            
            print(f"Download completed for folder '{folder_name}'")
            
        except HttpError as error:
            print(f'An error occurred: {error}')
    
    def load_transcripts(self):
        """Load and process transcript files from Max Life Videos folder"""
        # First, authenticate and choose folder
        service = self.authenticate()
        folders = self.list_drive_folders(service)
        
        if not folders:
            print("No folders found. Creating 'Max Life Videos' folder...")
            selected_folder = self.create_max_life_videos_folder(service)
            if not selected_folder:
                return
        else:
            selected_folder = self.choose_folder(folders)
            if not selected_folder:
                return
        
        # Download folder contents
        self.download_folder_contents(service, selected_folder['id'], selected_folder['name'])
        
        # Now process the downloaded files
        transcripts_folder = selected_folder['name']
        
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

class GoogleDriveSync:
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.json"):
        """
        Initialize Google Drive sync.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            token_file: Path to store authentication token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.local_sync_dir = "Max Life Videos"
        self.sync_interval = 30  # seconds
        self.is_watching = False
        self.watch_thread = None
        self.folder_id = None  # Will be set by find_or_create_folder()
        
        # Initialize Whisper model for transcription
        try:
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Whisper model: {e}")
            self.whisper_model = None
        
        # Authenticate with Google Drive
        try:
            self.service = self.authenticate()
            print("✅ Google Drive authentication successful")
        except Exception as e:
            print(f"❌ Google Drive authentication failed: {e}")
            raise
        
        # Create local sync directory if it doesn't exist
        os.makedirs(self.local_sync_dir, exist_ok=True)
        
    def authenticate(self):
        """Authenticate with Google Drive API."""
        creds = None
        
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception as e:
                print(f"⚠️ Error loading existing token: {e}")
                print("🗑️ Removing invalid token file...")
                os.remove('token.json')
                creds = None
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Error refreshing token: {e}")
                    print("🗑️ Removing invalid token file...")
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    creds = None
            
            if not creds:
                print("🔐 Starting Google Drive authentication...")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                    # Save the credentials for the next run
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                    print("✅ Authentication successful!")
                except Exception as e:
                    print(f"❌ Authentication failed: {e}")
                    raise
        
        return build('drive', 'v3', credentials=creds)
    
    def find_or_create_folder(self, folder_name: str = "Max Life Videos") -> str:
        """
        Find or create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to find/create
            
        Returns:
            Folder ID
        """
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        
        if files:
            self.folder_id = files[0]['id']
            print(f"✅ Found existing folder: {folder_name} (ID: {self.folder_id})")
            return self.folder_id
        
        # Create new folder if not found
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        self.folder_id = folder.get('id')
        print(f"✅ Created new folder: {folder_name} (ID: {self.folder_id})")
        return self.folder_id
    
    def list_drive_files(self) -> List[Dict[str, Any]]:
        """
        List all files in the Google Drive folder.
        
        Returns:
            List of file metadata dictionaries
        """
        if not self.folder_id:
            self.find_or_create_folder()
        
        query = f"'{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(
            q=query, 
            spaces='drive', 
            fields='files(id, name, mimeType, modifiedTime, size)',
            orderBy='modifiedTime desc'
        ).execute()
        
        return results.get('files', [])
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            local_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download {int(status.progress() * 100)}%")
            
            # Verify file integrity
            if not self._verify_file_integrity(local_path):
                print(f"⚠️ File integrity check failed for {os.path.basename(local_path)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading file: {e}")
            return False

    def _verify_file_integrity(self, file_path: str) -> bool:
        """
        Verify file integrity by checking if file is readable and has content.
        
        Args:
            file_path: Path to the file to verify
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check if file exists and has size > 0
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"⚠️ File is empty: {os.path.basename(file_path)}")
                return False
            
            # For video files, try to open with OpenCV to verify integrity
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in video_extensions:
                try:
                    import cv2
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        print(f"⚠️ Cannot open video file: {os.path.basename(file_path)}")
                        return False
                    cap.release()
                except ImportError:
                    # OpenCV not available, skip video verification
                    pass
                except Exception as e:
                    print(f"⚠️ Video file integrity check failed: {os.path.basename(file_path)} - {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ File integrity check error: {e}")
            return False
    
    def upload_file(self, local_path: str, drive_name: str = None) -> Optional[str]:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Local path of the file to upload
            drive_name: Name to use in Google Drive (defaults to local filename)
            
        Returns:
            Google Drive file ID if successful, None otherwise
        """
        if not self.folder_id:
            self.find_or_create_folder()
        
        if not drive_name:
            drive_name = os.path.basename(local_path)
        
        try:
            file_metadata = {
                'name': drive_name,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            print(f"✅ Uploaded {drive_name} (ID: {file.get('id')})")
            return file.get('id')
        except HttpError as error:
            print(f"❌ Error uploading file {local_path}: {error}")
            return None
    
    def sync_from_drive(self) -> Dict[str, List[str]]:
        """
        Sync files from Google Drive to local directory.
        
        Returns:
            Dictionary with 'downloaded', 'updated', and 'errors' lists
        """
        drive_files = self.list_drive_files()
        local_files = set(os.listdir(self.local_sync_dir)) if os.path.exists(self.local_sync_dir) else set()
        
        downloaded = []
        updated = []
        errors = []
        
        # Create a mapping of base names to check for existing transcripts
        base_name_map = {}
        for local_file in local_files:
            base_name = os.path.splitext(local_file)[0]
            if base_name not in base_name_map:
                base_name_map[base_name] = []
            base_name_map[base_name].append(local_file)
        
        for file in drive_files:
            file_name = file['name']
            local_path = os.path.join(self.local_sync_dir, file_name)
            
            # Check if file needs to be downloaded or updated
            should_download = False
            if file_name not in local_files:
                should_download = True
                downloaded.append(file_name)
            else:
                # Check if remote file is newer
                try:
                    local_mtime = os.path.getmtime(local_path) if os.path.exists(local_path) else 0
                    drive_mtime = datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00')).timestamp()
                    
                    if drive_mtime > local_mtime:
                        should_download = True
                        updated.append(file_name)
                except Exception as e:
                    print(f"⚠️ Error checking file timestamps for {file_name}: {e}")
                    # If we can't check timestamps, download to be safe
                    should_download = True
                    updated.append(file_name)
            
            if should_download:
                try:
                    # Create a temporary file first to avoid corruption
                    temp_path = local_path + '.tmp'
                    if self.download_file(file['id'], temp_path):
                        # Move temp file to final location
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        os.rename(temp_path, local_path)
                        print(f"✅ Synced: {file_name}")
                    else:
                        errors.append(file_name)
                        # Clean up temp file if it exists
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                except Exception as e:
                    print(f"❌ Error syncing {file_name}: {e}")
                    errors.append(file_name)
                    # Clean up temp file if it exists
                    temp_path = local_path + '.tmp'
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        return {
            'downloaded': downloaded,
            'updated': updated,
            'errors': errors
        }
    
    def transcribe_video(self, video_path: str) -> bool:
        """
        Transcribe a video file using Whisper.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🎵 Transcribing: {os.path.basename(video_path)}")
            
            # Transcribe the video
            result = self.whisper_model.transcribe(video_path)
            
            # Save transcript as text file
            base_name = os.path.splitext(video_path)[0]
            txt_path = f"{base_name}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for segment in result['segments']:
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text'].strip()
                    f.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")
            
            print(f"✅ Transcription saved: {os.path.basename(txt_path)}")
            
            # Also generate SRT file
            self.generate_srt_file(video_path, result)
            
            return True
            
        except Exception as e:
            print(f"❌ Error transcribing {video_path}: {e}")
            return False
    
    def generate_srt_file(self, video_path: str, result: dict = None) -> bool:
        """
        Generate SRT subtitle file from video transcription.
        
        Args:
            video_path: Path to the video file
            result: Whisper transcription result (if None, will transcribe)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_name = os.path.splitext(video_path)[0]
            srt_path = f"{base_name}.srt"
            
            # If no result provided, transcribe the video
            if result is None:
                result = self.whisper_model.transcribe(video_path)
            
            # Generate SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    start_time = self._format_srt_time(segment['start'])
                    end_time = self._format_srt_time(segment['end'])
                    text = segment['text'].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            print(f"✅ SRT file saved: {os.path.basename(srt_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating SRT for {video_path}: {e}")
            return False

    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def process_new_files(self, files: List[str]):
        """
        Process new files by transcribing videos and uploading transcripts/SRTs.
        Only transcribe videos that don't already have transcript files.
        
        Args:
            files: List of new file names
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        for file_name in files:
            file_path = os.path.join(self.local_sync_dir, file_name)
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext in video_extensions:
                print(f"🎬 Processing video: {file_name}")
                
                # Check if transcript files already exist
                base_name = os.path.splitext(file_path)[0]
                txt_path = f"{base_name}.txt"
                srt_path = f"{base_name}.srt"
                
                txt_exists = os.path.exists(txt_path)
                srt_exists = os.path.exists(srt_path)
                
                # Check if transcript files exist in Google Drive as well
                drive_files = self.list_drive_files()
                drive_file_names = [f['name'] for f in drive_files]
                txt_name = os.path.basename(txt_path)
                srt_name = os.path.basename(srt_path)
                
                txt_in_drive = txt_name in drive_file_names
                srt_in_drive = srt_name in drive_file_names
                
                # Only transcribe if no transcript files exist locally or in Drive
                if not txt_exists and not srt_exists and not txt_in_drive and not srt_in_drive:
                    print(f"🎵 Transcribing video (no existing transcripts found): {file_name}")
                    
                    # Transcribe the video
                    if self.transcribe_video(file_path):
                        # Upload transcript to Google Drive
                        if os.path.exists(txt_path):
                            txt_name = os.path.basename(txt_path)
                            self.upload_file(txt_path, txt_name)
                            print(f"📤 Uploaded transcript: {txt_name}")
                        
                        # Generate and upload SRT file
                        if self.generate_srt_file(file_path, result=None):
                            srt_path = f"{base_name}.srt"
                            if os.path.exists(srt_path):
                                srt_name = os.path.basename(srt_path)
                                self.upload_file(srt_path, srt_name)
                                print(f"📤 Uploaded SRT: {srt_name}")
                else:
                    print(f"⏭️ Skipping transcription for {file_name} - transcript files already exist")
                    if txt_exists:
                        print(f"   📄 Local transcript: {txt_name}")
                    if srt_exists:
                        print(f"   📄 Local SRT: {srt_name}")
                    if txt_in_drive:
                        print(f"   ☁️ Drive transcript: {txt_name}")
                    if srt_in_drive:
                        print(f"   ☁️ Drive SRT: {srt_name}")
                    
                    # Upload existing local files to Drive if they're not there
                    if txt_exists and not txt_in_drive:
                        self.upload_file(txt_path, txt_name)
                        print(f"📤 Uploaded existing transcript: {txt_name}")
                    
                    if srt_exists and not srt_in_drive:
                        self.upload_file(srt_path, srt_name)
                        print(f"📤 Uploaded existing SRT: {srt_name}")
    
    def start_watching(self):
        """Start watching for new files in Google Drive."""
        if self.is_watching:
            print("⚠️ Already watching for changes")
            return
        
        self.is_watching = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        print(f"👀 Started watching Google Drive for changes (checking every {self.sync_interval}s)")
    
    def stop_watching(self):
        """Stop watching for new files."""
        self.is_watching = False
        if self.watch_thread:
            self.watch_thread.join()
        print("⏹️ Stopped watching Google Drive")
    
    def _watch_loop(self):
        """Main loop for watching Google Drive changes."""
        last_sync = {}
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while self.is_watching:
            try:
                # Get current files from Google Drive
                drive_files = self.list_drive_files()
                current_files = {f['name']: f['modifiedTime'] for f in drive_files}
                
                # Check for new or modified files
                new_files = []
                for file_name, modified_time in current_files.items():
                    if file_name not in last_sync or last_sync[file_name] != modified_time:
                        new_files.append(file_name)
                
                if new_files:
                    print(f"🔄 Found {len(new_files)} new/modified files: {new_files}")
                    
                    # Sync files from Google Drive
                    sync_result = self.sync_from_drive()
                    
                    # Process new files (transcribe videos)
                    all_new = sync_result['downloaded'] + sync_result['updated']
                    if all_new:
                        self.process_new_files(all_new)
                    
                    # Update last sync times
                    for file_name in new_files:
                        if file_name in current_files:
                            last_sync[file_name] = current_files[file_name]
                    
                    # Reset error counter on successful sync
                    consecutive_errors = 0
                
                # Wait before next check
                time.sleep(self.sync_interval)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error in watch loop (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"⚠️ Too many consecutive errors ({consecutive_errors}), pausing sync for 60 seconds...")
                    time.sleep(60)
                    consecutive_errors = 0
                else:
                    # Exponential backoff for errors
                    backoff_time = min(self.sync_interval * (2 ** consecutive_errors), 60)
                    print(f"⏳ Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
    
    def initial_sync(self):
        """Perform initial sync from Google Drive."""
        print("🔄 Performing initial sync from Google Drive...")
        
        # Clean up any interrupted sync operations
        self._cleanup_interrupted_sync()
        
        sync_result = self.sync_from_drive()
        
        print(f"📥 Downloaded: {len(sync_result['downloaded'])} files")
        print(f"🔄 Updated: {len(sync_result['updated'])} files")
        if sync_result['errors']:
            print(f"❌ Errors: {len(sync_result['errors'])} files")
        
        # Process any new files
        all_new = sync_result['downloaded'] + sync_result['updated']
        if all_new:
            self.process_new_files(all_new)
        
        return sync_result

    def _cleanup_interrupted_sync(self):
        """Clean up any temporary files from interrupted sync operations."""
        if not os.path.exists(self.local_sync_dir):
            return
        
        print("🧹 Cleaning up interrupted sync operations...")
        cleaned_count = 0
        
        for file_name in os.listdir(self.local_sync_dir):
            if file_name.endswith('.tmp'):
                temp_path = os.path.join(self.local_sync_dir, file_name)
                try:
                    os.remove(temp_path)
                    cleaned_count += 1
                    print(f"🗑️ Removed temp file: {file_name}")
                except Exception as e:
                    print(f"⚠️ Could not remove temp file {file_name}: {e}")
        
        if cleaned_count > 0:
            print(f"✅ Cleaned up {cleaned_count} temporary files")
        else:
            print("✅ No temporary files found")

    def sync_folder(self) -> Dict[str, List[str]]:
        """
        Sync the entire folder from Google Drive.
        This is a convenience method that combines cleanup and sync.
        
        Returns:
            Dictionary with sync results
        """
        print("🔄 Starting folder sync...")
        
        # Clean up any interrupted operations
        self._cleanup_interrupted_sync()
        
        # Perform sync
        return self.sync_from_drive()

# Example usage and setup
def setup_google_drive_sync():
    """
    Setup function to initialize Google Drive sync.
    Call this before using the video processing pipeline.
    """
    try:
        # Initialize Google Drive sync
        drive_sync = GoogleDriveSync()
        
        # Perform initial sync
        drive_sync.initial_sync()
        
        # Start watching for changes
        drive_sync.start_watching()
        
        return drive_sync
        
    except Exception as e:
        print(f"❌ Failed to setup Google Drive sync: {e}")
        return None

if __name__ == "__main__":
    print("=== Google Drive Sync for ClipQuery ===")
    print("This script will help you sync your video transcripts from Google Drive.")
    print("You can choose an existing folder or create a new 'Max Life Videos' folder.")
    print()
    
    # Initialize the vector store and start the sync process
    vector_store = VectorStore()
    vector_store.load_transcripts()
    
    print("\n=== Sync Complete ===")
    print("Your video transcripts have been downloaded and processed.")
    print("You can now use your ClipQuery application with the synced data.")
    print("\nTo stop the sync process, press Ctrl+C")
    
    # Keep the script running to maintain the sync
    try:
        while True:
            import time
            time.sleep(60)  # Check for updates every minute
            print("Checking for new files...")
            # You could add logic here to check for new files periodically
    except KeyboardInterrupt:
        print("\nSync stopped by user.") 