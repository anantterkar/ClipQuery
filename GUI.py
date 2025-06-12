import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import time
from datetime import datetime


os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Import functions from audio_analysis
from audio_analysis import (
    download_video_from_url,
    convert_to_mp4,
    extract_audio,
    extract_subtitles,
    transcribe_audio
)

# import ollama related functions here
from langchain_ollama import (
    answer_query_with_ollama,
    get_major_topics,
    find_most_similar_topic,
    find_similar_transcript_segments,
    clip_video,
    general_template,
    chain_general,
    prompt_general,
)
    
    

class ChatMessage:
    def __init__(self, text, is_user=True, timestamp=None):
        self.text = text
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now()

class VideoSnipperChatbot:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Vivi - Video Vision")
        self.root.geometry("900x700")
        
        # Set background color to light pink
        self.root.configure(fg_color="#FFE6E6")
        
        # Variables
        self.input_path = ""
        self.output_dir = r"C:\Users\adibr\Desktop\video_clip\video_folder"
        self.query = ""
        self.is_processing = False
        self.conversation_step = "greeting"  # greeting, input, query, processing, complete
        self.transcript_ready = False
        self.transcript_path = None
        self.setup_ui()
        self.start_conversation()
        
    def setup_ui(self):
        # Configure grid weight
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Enable drag and drop
        #self.setup_drag_drop()
        
        # Header
        header_frame = ctk.CTkFrame(self.root, height=80, corner_radius=0, fg_color="#FFE6E6")
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        
        # Bot avatar and title
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.pack(expand=True, fill="both", padx=20)
        
        bot_avatar = ctk.CTkLabel(
            header_content, 
            text="🤖", 
            font=ctk.CTkFont(size=32)
        )
        bot_avatar.pack(side="left", pady=20)
        
        title_frame = ctk.CTkFrame(header_content, fg_color="transparent")
        title_frame.pack(side="left", padx=(15, 0), pady=20)
        
        bot_name = ctk.CTkLabel(
            title_frame, 
            text="Vivi - Video Vision", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        bot_name.pack(anchor="w")
        
        bot_status = ctk.CTkLabel(
            title_frame, 
            text="Online • Ready to help", 
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        bot_status.pack(anchor="w")
        
        # Chat area
        self.chat_frame = ctk.CTkScrollableFrame(self.root, fg_color="#FFE6E6")
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Input area
        input_frame = ctk.CTkFrame(self.root, height=120, corner_radius=15, fg_color="#FFE6E6")
        input_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_frame.grid_propagate(False)
        input_frame.grid_columnconfigure(1, weight=1)
        
        # File drop area (initially hidden)
        self.drop_area = ctk.CTkFrame(
            input_frame, 
            height=60,
            fg_color=("gray90", "gray20"),
            border_width=2,
            border_color=("gray70", "gray50")
        )
        
        self.drop_label = ctk.CTkLabel(
            self.drop_area,
            text="📁 Drop video file here or click to browse",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        self.drop_label.pack(expand=True)
        
        # Bind click event to drop area
        self.drop_area.bind("<Button-1>", lambda e: self.browse_file_dialog())
        self.drop_label.bind("<Button-1>", lambda e: self.browse_file_dialog())
        
        # Message input
        self.message_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message...",
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=20
        )
        self.message_entry.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=20)
        self.message_entry.bind("<Return>", self.send_message)
        
        # Quick action buttons
        self.quick_buttons_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        self.quick_buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            height=35,
            corner_radius=15,
            fg_color="#00205B",  # Navy blue color for send button
            hover_color="#000066"  # Darker navy blue for hover
        )
        self.send_btn.grid(row=0, column=2, padx=(10, 20), pady=20)
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        try:
            import tkinterdnd2 as tkdnd
            self.root = tkdnd.Tk()  # Replace CTk with DND-enabled Tk
            # Note: This would require tkinterdnd2 package
        except ImportError:
            # Fallback: Use basic file handling
            pass
            
    def show_file_drop_area(self):
        """Show the file drop area"""
        self.drop_area.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20, pady=(0, 10))
        
    def hide_file_drop_area(self):
        """Hide the file drop area"""
        self.drop_area.grid_remove()
        
    def browse_file_dialog(self):
        """Open file dialog"""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filename:
            self.handle_file_selection(filename)
            
    def handle_file_selection(self, filepath):
        """Handle file selection from any method"""
        if not os.path.exists(filepath):
            error_msg = "❌ File not found. Please try selecting the file again."
            typing = self.add_typing_indicator()
            self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
            return
            
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext not in valid_extensions:
            error_msg = f"❌ Unsupported file format: {file_ext}\n\nSupported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"
            typing = self.add_typing_indicator()
            self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
            return
            
        # File is valid
        self.input_path = filepath
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)
        
        file_msg = f"📁 {filename}\n💾 Size: {file_size_mb:.1f} MB"
        self.create_message_bubble(file_msg, is_user=True)
        
        self.hide_quick_buttons()
        self.hide_file_drop_area()
        
        # Proceed to next step
        success_msg = f"Perfect! I've got your video file: {filename}"
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, success_msg))
        self.root.after(1500, self.ask_for_query)
        
    def create_message_bubble(self, message, is_user=True):
        """Create a message bubble widget"""
        # Container for the entire message
        msg_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        msg_container.grid(row=len(self.chat_frame.winfo_children()), column=0, sticky="ew", padx=10, pady=5)
        msg_container.grid_columnconfigure(0 if not is_user else 2, weight=1)
        chat_font = 20
        
        if is_user:
            # User message (right-aligned, navy blue)
            msg_frame = ctk.CTkFrame(
                msg_container, 
                fg_color="#00205B",  # Navy blue color for user messages
                corner_radius=15
            )
            msg_frame.grid(row=0, column=1, sticky="e", padx=(80, 0))
            
            msg_label = ctk.CTkLabel(
                msg_frame,
                text=message,
                font=ctk.CTkFont(size=chat_font),
                wraplength=400,
                justify="left",
                text_color="white"
            )
            msg_label.pack(padx=15, pady=10)
            
            # User avatar
            avatar = ctk.CTkLabel(msg_container, text="👤", font=ctk.CTkFont(size=20))
            avatar.grid(row=0, column=2, padx=(10, 0), sticky="ne")
            
        else:
            # Bot message (left-aligned, burgundy)
            # Bot avatar
            avatar = ctk.CTkLabel(msg_container, text="🤖", font=ctk.CTkFont(size=20))
            avatar.grid(row=0, column=0, padx=(0, 10), sticky="nw")
            
            msg_frame = ctk.CTkFrame(
                msg_container, 
                fg_color="#7F1734",  # Burgundy color for bot messages
                corner_radius=15
            )
            msg_frame.grid(row=0, column=1, sticky="w", padx=(0, 80))
            
            msg_label = ctk.CTkLabel(
                msg_frame,
                text=message,
                font=ctk.CTkFont(size=chat_font),
                wraplength=400,
                justify="left",
                text_color="white"
            )
            msg_label.pack(padx=15, pady=10)
        
        # Scroll to bottom
        self.root.update_idletasks()
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
    def add_typing_indicator(self):
        """Add typing indicator for bot"""
        typing_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        typing_container.grid(row=len(self.chat_frame.winfo_children()), column=0, sticky="ew", padx=10, pady=5)
        typing_container.grid_columnconfigure(2, weight=1)
        
        avatar = ctk.CTkLabel(typing_container, text="🤖", font=ctk.CTkFont(size=20))
        avatar.grid(row=0, column=0, padx=(0, 10), sticky="nw")
        
        typing_frame = ctk.CTkFrame(
            typing_container, 
            fg_color=("#f0f0f0", "#2b2b2b"),
            corner_radius=15
        )
        typing_frame.grid(row=0, column=1, sticky="w")
        
        typing_label = ctk.CTkLabel(
            typing_frame,
            text="• • •",
            font=ctk.CTkFont(size=13),
            text_color=("gray", "gray60")
        )
        typing_label.pack(padx=15, pady=10)
        
        self.root.update_idletasks()
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
        return typing_container
        
    def remove_typing_indicator(self, typing_widget):
        """Remove typing indicator"""
        typing_widget.destroy()
        
    def show_quick_buttons(self, buttons):
        """Show quick action buttons"""
        # Clear existing buttons
        for widget in self.quick_buttons_frame.winfo_children():
            widget.destroy()
            
        for i, (text, command) in enumerate(buttons):
            btn = ctk.CTkButton(
                self.quick_buttons_frame,
                text=text,
                command=command,
                height=30,
                corner_radius=15,
                font=ctk.CTkFont(size=12),
                fg_color="#00205B",  # Navy blue color for quick buttons
                hover_color="#000066"  # Darker navy blue for hover
            )
            btn.pack(side="left", padx=5)
            
    def hide_quick_buttons(self):
        """Hide quick action buttons"""
        for widget in self.quick_buttons_frame.winfo_children():
            widget.destroy()
            
    def start_conversation(self):
        """Start the conversation with greeting"""
        greeting_msg = """Hi there! 👋 I'm Vivi, your Video Vision AI Assistant. 

I can help you process videos and create transcriptions with AI-powered analysis.

Would you like to:
1. Query videos from our existing database
2. Upload and process a new video"""
        
        self.create_message_bubble(greeting_msg, is_user=False)
        
        # Show quick buttons for initial choice
        quick_actions = [
            ("📚 Query Database", self.query_database),
            ("📤 Upload New Video", self.upload_new_video),
            ("❓ Learn More", self.show_help)
        ]
        self.show_quick_buttons(quick_actions)
        
    def query_database(self):
        """Handle querying existing videos from database"""
        self.create_message_bubble("📚 Query Database", is_user=True)
        self.hide_quick_buttons()
        
        db_msg = """Please select a video from the database folder"""
        
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, db_msg))
        
        # Open file dialog to browse video folder
        video_folder = r"C:\Users\adibr\Desktop\video_clip\video_folder"
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video from Database",
            initialdir=video_folder,
            filetypes=filetypes
        )
        
        if filename:
            # Get just the filename without the path
            video_file = os.path.basename(filename)
            self.handle_video_selection(video_file)
        else:
            # User cancelled the selection
            cancel_msg = "No video selected. Would you like to try again?"
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, cancel_msg))
            
            retry_actions = [
                ("🔄 Try Again", self.query_database),
                ("⬅️ Back to Main Menu", self.restart_conversation)
            ]
            self.root.after(1500, lambda: self.show_quick_buttons(retry_actions))

    def handle_video_selection(self, video_file):
        """Handle the selection of a video from the database"""
        self.create_message_bubble(f"Selected: {video_file}", is_user=True)
        
        # Get the video path
        video_path = os.path.join(r"C:\Users\adibr\Desktop\video_clip\video_folder", video_file)
        
        # Look for corresponding transcript in transcripts folder
        transcript_file = os.path.splitext(video_file)[0] + ".txt"
        self.transcript_path = os.path.join(r"C:\Users\adibr\Desktop\video_clip\transcripts", transcript_file)
        
        if os.path.exists(self.transcript_path):
            # Read the transcript
            with open(self.transcript_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            
            # Show ready message
            ready_msg = "✨ Vivi is ready to answer any queries about the video selected from the database!"
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, ready_msg))
            
            # Set the input path for future queries
            self.input_path = video_path
            self.transcript_ready = True
            # Ask for query
            self.root.after(1500, self.ask_for_query)
        else:
            error_msg = f"❌ No transcript found for {video_file} in the transcripts folder. Would you like to process this video now?"
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, error_msg))
            
            process_actions = [
                ("🔄 Process Video", lambda: self.start_processing()),
                ("⬅️ Back to List", self.query_database)
            ]
            self.root.after(1500, lambda: self.show_quick_buttons(process_actions))

    def upload_new_video(self):
        """Handle uploading a new video"""
        self.create_message_bubble("📤 Upload New Video", is_user=True)
        self.hide_quick_buttons()
        
        # Show file drop area and instructions
        file_msg = "Great! You can provide your video file in several ways:"
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, file_msg))
        
        # Show file drop area
        self.root.after(1200, self.show_file_selection_options)

    def show_file_selection_options(self):
        """Show file selection options"""
        self.show_file_drop_area()
        
        options_msg = """📁 **File Selection Options:**
        Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"""
        
        typing = self.add_typing_indicator()
        self.root.after(500, lambda: self.remove_typing_and_reply(typing, options_msg))
        
        # Set conversation state
        self.conversation_step = "awaiting_file"
        
        # Show quick browse button
        browse_actions = [("📂 Browse Files", self.browse_file_dialog)]
        self.root.after(1000, lambda: self.show_quick_buttons(browse_actions))
        
    def show_help(self):
        """Show help information"""
        self.create_message_bubble("❓ Learn More", is_user=True)
        self.hide_quick_buttons()
        
        help_msg = """Here's what I can do for you:

📚 **Query Database**: Search and analyze videos already in our system
📤 **Upload New Videos**: Add new videos to our database for processing
✨ **Process Video Links**: Download and transcribe YouTube videos
📁 **Process Local Videos**: Upload your own MP4, AVI, MOV, or other video files
🧠 **Smart Analysis**: Add custom queries to focus on specific topics
📝 **Multiple Formats**: Get clean transcripts saved to your chosen directory

Ready to get started? Just let me know what you'd like to do!"""
        
        # Simulate typing delay
        typing = self.add_typing_indicator()
        self.root.after(1500, lambda: self.remove_typing_and_reply(typing, help_msg))
        
        # Show main options again
        quick_actions = [
            ("📚 Query Database", self.query_database),
            ("📤 Upload New Video", self.upload_new_video)
        ]
        self.root.after(2000, lambda: self.show_quick_buttons(quick_actions))
        
    def remove_typing_and_reply(self, typing_widget, message):
        """Remove typing indicator and show message"""
        self.remove_typing_indicator(typing_widget)
        self.create_message_bubble(message, is_user=False)
        
        # Show main options again
        quick_actions = [
            ("📚 Query Database", self.query_database),
            ("📤 Upload New Video", self.upload_new_video)
        ]
        self.show_quick_buttons(quick_actions)
        
    def ask_for_youtube_url(self):
        """Ask for YouTube URL"""
        msg = "Great! Please paste the YouTube URL you'd like me to process:"
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, msg))
        self.conversation_step = "awaiting_url"
        
    def browse_local_file(self):
        """Browse for local file"""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filename:
            self.input_path = filename
            file_msg = f"Selected file: {os.path.basename(filename)}"
            self.create_message_bubble(file_msg, is_user=True)
            #self.ask_for_query()
        else:
            msg = "No file selected. Would you like to try again?"
            typing = self.add_typing_indicator()
            self.root.after(500, lambda: self.remove_typing_and_reply(typing, msg))
            
            retry_actions = [
                ("📁 Browse Again", self.quick_local_file),
                ("📹 YouTube Instead", self.quick_youtube)
            ]
            self.show_quick_buttons(retry_actions)
            
    def ask_for_query(self):
        """Ask for optional query"""
        msg = """Please type in your queries"""
        
        typing = self.add_typing_indicator()
        self.root.after(1000, lambda: self.remove_typing_and_reply(typing, msg))
        self.conversation_step = "awaiting_query"
        
        skip_actions = [("⏭️ Skip Query", self.skip_query)]
        self.show_quick_buttons(skip_actions)
        
    def skip_query(self):
        """Skip query and proceed"""
        self.create_message_bubble("⏭️ Skip Query", is_user=True)
        self.query = ""
        self.hide_quick_buttons()
        self.start_processing()
        
    def send_message(self, event=None):
        """Handle sending messages"""
        message = self.message_entry.get().strip()
        if not message:
            return
            
        self.create_message_bubble(message, is_user=True)
        self.message_entry.delete(0, 'end')
        self.hide_quick_buttons()
        
        # Handle based on conversation step
        if self.conversation_step == "awaiting_url":
            self.handle_url_input(message)
        elif self.conversation_step == "awaiting_query":
            self.handle_query_input(message)
        elif self.conversation_step == "awaiting_file":
            self.handle_file_path_input(message)
        else:
            # General conversation
            self.handle_general_message(message)
            
    def handle_file_path_input(self, filepath):
        """Handle file path input from message"""
        # Clean up the path (remove quotes if present)
        filepath = filepath.strip('\'"')
        
        # Check if it's a valid file path
        if os.path.exists(filepath) and os.path.isfile(filepath):
            self.handle_file_selection(filepath)
        else:
            error_msg = f"❌ I couldn't find a file at: {filepath}\n\nPlease check the path and try again, or use the browse button above."
            typing = self.add_typing_indicator()
            self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
            
    def handle_url_input(self, url):
        """Handle YouTube URL input"""
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            self.input_path = url
            msg = f"Got it! I'll process this YouTube video: {url[:50]}..."
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, msg))
            self.root.after(1500, self.ask_for_query)
        else:
            error_msg = "That doesn't look like a YouTube URL. Please make sure it contains 'youtube.com' or 'youtu.be'"
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, error_msg))
            
    


    def handle_query_input(self, query):
        """Handle query input"""
        if query.lower() in ["skip", "no", "none", ""]:
            self.query = ""
            msg = "No problem! I'll do a standard transcription."
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, msg))
            # ...existing logic...
            return

        self.query = query
        msg = f"Ok, your query is being analyzed by Vivi"
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, msg))

        def run_query():
            try:
                # --- Video Clipping Query ---
                if query.lower().startswith("video clipping:"):
                    # Extract the actual query
                    clip_query = query[len("video clipping:"):].strip()
                    topics = get_major_topics(self.full_transcript_text)
                    if not topics or "Error" in topics[0]["description"]:
                        response = topics[0]["description"]
                    else:
                        best_topic, similarity = find_most_similar_topic(clip_query, topics)
                        if not best_topic or similarity < 0.1:
                            response = "No relevant topic found for the video."
                        else:
                            result = clip_video(
                                self.input_path,
                                best_topic["start"],
                                best_topic["end"],
                                output_path=f"clipped_{int(best_topic['start'])}_{int(best_topic['end'])}.mp4"
                            )
                            response = f"{result}\nTopic: {best_topic['description']}\nTimestamps: {best_topic['start']:.2f} - {best_topic['end']:.2f}\nSimilarity: {similarity:.2f}"
                    self.root.after(0, lambda: self.create_message_bubble(response, is_user=False))
                else:
                    # --- General Query ---
                    # Optionally, show relevant transcript segments
                    top_segments = find_similar_transcript_segments(query, self.transcript_segments, top_k=3)
                    answer = answer_query_with_ollama(self.full_transcript_text, query, self.context)
                    self.root.after(0, lambda: self.create_message_bubble(answer, is_user=False))
                    if top_segments:
                        seg_text = "Relevant segments:\n" + "\n".join(
                            [f"[{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}" for seg in top_segments])
                        self.root.after(0, lambda: self.create_message_bubble(seg_text, is_user=False))
                    # Update conversation context
                    self.context += f"\nUser: {query}\nAI: {answer}\n"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self.create_message_bubble(error_msg, is_user=False))

        threading.Thread(target=run_query, daemon=True).start()

    def handle_general_message(self, message):
        """Handle general conversation"""
        # Simple keyword-based responses
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            response = "Hello! How can I help you with video processing today?"
        elif any(word in message_lower for word in ["help", "what", "how"]):
            response = "I can process YouTube videos and local video files to create transcriptions. Would you like to get started?"
        else:
            response = "I'd be happy to help you process a video! Would you like to work with a YouTube video or a local file?"
            
        typing = self.add_typing_indicator()
        self.root.after(1000, lambda: self.remove_typing_and_reply(typing, response))
        
        # Show options
        quick_actions = [
            ("📹 YouTube Video", self.quick_youtube),
            ("📁 Local File", self.quick_local_file)
        ]
        self.root.after(1500, lambda: self.show_quick_buttons(quick_actions))
        
    def start_processing(self):
        """Start processing with chatbot updates"""
        if not self.input_path:
            error_msg = "Oops! I don't have a video to process. Let's start over."
            typing = self.add_typing_indicator()
            self.root.after(800, lambda: self.remove_typing_and_reply(typing, error_msg))
            self.root.after(1500, self.start_conversation)
            return
            
        start_msg = "🚀 Starting the processing! This might take a few minutes..."
        typing = self.add_typing_indicator()
        self.root.after(1000, lambda: self.remove_typing_and_reply(typing, start_msg))
        
        # Start processing in thread
        self.root.after(1500, lambda: threading.Thread(target=self.process_video, daemon=True).start())
        
    def process_video(self):
        """Process video with chatbot updates"""
        try:
            self.is_processing = True
            
            # Validate input path
            if not self.input_path:
                error_msg = "❌ No video file selected. Please select a video first."
                typing = self.add_typing_indicator()
                self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
                self.root.after(1500, self.restart_conversation)
                return
                
            if not os.path.exists(self.input_path):
                error_msg = f"❌ Video file not found: {self.input_path}"
                typing = self.add_typing_indicator()
                self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
                self.root.after(1500, self.restart_conversation)
                return
            
            # Step 1: Video extraction
            extract_msg = "🎬 Extracting audio from video..."
            typing = self.add_typing_indicator()
            self.root.after(500, lambda: self.remove_typing_and_reply(typing, extract_msg))
            
            try:
                
                video_path = None
                print(self.input_path)
                
                # If input is a URL, download it first
                if isinstance(self.input_path, str) and self.input_path.startswith(('http://', 'https://')):
                    video_path = download_video_from_url(self.input_path, os.path.join(self.output_dir, "input_video.mp4"))
                elif os.path.exists(self.input_path):
                    video_path = self.input_path
                else:
                    # Convert to MP4 if needed
                    video_path = convert_to_mp4(self.input_path, os.path.join(self.output_dir, "input_video.mp4"))

                # Extract audio using the extract_audio function
                audio_path = os.path.join(self.output_dir, "audio.wav")
                extract_audio(video_path, audio_path)
                success_msg = "✅ Audio extraction completed!"

                typing = self.add_typing_indicator()
                self.root.after(1000, lambda: self.remove_typing_and_reply(typing, success_msg))
            except Exception as e:
                error_msg = f"❌ Failed to extract audio: {str(e)}"
                typing = self.add_typing_indicator()
                self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
                raise e
                
            # Step 2: Transcription
            transcribe_msg = "🎤 Transcribing audio with AI..."
            typing = self.add_typing_indicator()
            self.root.after(1000, lambda: self.remove_typing_and_reply(typing, transcribe_msg))
            
            try:
                # Always transcribe first without query
                self.transcript_path = transcribe_audio(audio_path, self.output_dir)
                    
                complete_msg = f"""🎉 Transcription completed!

                📁 Files saved to: {self.output_dir}
                📝 Transcript: {os.path.basename(self.transcript_path)}"""
                    
                typing = self.add_typing_indicator()
                self.root.after(1500, lambda: self.remove_typing_and_reply(typing, complete_msg))
                
                # Show ready message
                ready_msg = "✨ Vivi is ready to answer any queries about the video!"
                typing = self.add_typing_indicator()
                self.root.after(1000, lambda: self.remove_typing_and_reply(typing, ready_msg))
                
                # Show query input
                self.root.after(1500, self.ask_for_query)
                
            except Exception as e:
                error_msg = f"❌ Transcription failed: {str(e)}"
                typing = self.add_typing_indicator()
                self.root.after(500, lambda: self.remove_typing_and_reply(typing, error_msg))
                raise e
                
        except Exception as e:
            final_error = f"❌ Sorry, something went wrong: {str(e)}\n\nWould you like to try again?"
            typing = self.add_typing_indicator()
            self.root.after(1000, lambda: self.remove_typing_and_reply(typing, final_error))
            
            retry_actions = [("🔄 Try Again", self.restart_conversation)]
            self.root.after(1500, lambda: self.show_quick_buttons(retry_actions))
            
        finally:
            self.is_processing = False
            
    def restart_conversation(self):
        """Restart the conversation"""
        self.create_message_bubble("🔄 Process Another Video", is_user=True)
        self.hide_quick_buttons()
        self.hide_file_drop_area()
        
        # Reset variables
        self.input_path = ""
        self.query = ""
        self.conversation_step = "greeting"
        
        restart_msg = "Great! Let's process another video. What would you like to do?"
        typing = self.add_typing_indicator()
        self.root.after(800, lambda: self.remove_typing_and_reply(typing, restart_msg))
        
        # Show main options
        quick_actions = [
            ("📹 YouTube Video", self.quick_youtube),
            ("📁 Local File", self.quick_local_file)
        ]
        self.root.after(1500, lambda: self.show_quick_buttons(quick_actions))
        
    def run(self):
        """Start the application"""
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"900x700+{x}+{y}")
        
        self.root.mainloop()

def main():
    app = VideoSnipperChatbot()
    app.run()

if __name__ == "__main__":
    main()