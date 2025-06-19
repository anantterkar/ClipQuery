import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
from datetime import datetime
from PIL import Image, ImageTk
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# import functions of python files
# from version_2 import (
#     download_video_from_url,
#     convert_to_mp4,
#     extract_audio,
#     extract_subtitles,
#     transcribe_audio,
#     answer_query_with_ollama,
#     get_major_topics,
#     find_most_similar_topic,
#     find_similar_transcript_segments,
#     clip_video,
#     general_template,
#     chain_general,
#     prompt_general,
# )

class VideoSnipperChatbot:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Vivi - Video Vision")
        self.root.geometry("900x700")
        self.root.configure(fg_color = "#FFE6E6")
        
        #input variables
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
        
        # logo_path = "clipquery_logo.jpeg"
        # logo_image = Image.open(logo_path)
        # logo_image = logo_image.resize((120, 0))
        # self.logo_photo = ImageTk.PhotoImage(logo_image)
        
        # logo_label = ctk.CTkLabel(header_frame, image = self.logo_photo)
        # logo_label.pack(side='left', padx=(10,10), pady=10)
        
    def setup_drag_drop(self):
        # for drag and drop functionality
        import tkinterdnd2 as tkdnd
        self.root = tkdnd.Tk()
    
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
        
        if not os.path.exists(filepath):
            error_msg = "File not found. Please try selecting the correct file"    
            return
        
        # check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        file_ext = os.path.splitext(filepath)[1].lower()

        if file_ext not in valid_extensions:
            error_msg = f"Unsupported file format"
            self.create_message_bubble(error_msg, is_user=False)
            return
        
        self.input_path = filepath
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size/(1024*1024)
        
        file_msg = f"📁 {filename}\n💾 Size: {file_size_mb:.1f} MB"
        self.create_message_bubble(file_msg, is_user=False)
        
        self.hide_quick_button()
        self.hide_file_drop_area()
        
        success_msg = f"Perfect! I've got your video file: {filename}"
        self.create_message_bubble(success_msg, is_user=False)
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
        
        self.create_message_bubble(help_msg, is_user=False)
        # Show main options again
        quick_actions = [
            ("📚 Query Database", self.query_database),
            ("📤 Upload New Video", self.upload_new_video)
        ]
        self.root.after(2000, lambda: self.show_quick_buttons(quick_actions))
        
    def start_conversation(self):
        greeting_msg = """Hi there! 👋 I'm Vivi, your Video Vision AI Assistant. I can help you process videos and create transcriptions with AI-powered analysis. Would you like to:
        1. Query videos from our existing database
        2. Upload and process a new video"""
        
        self.create_message_bubble(greeting_msg, is_user=False)
        
        quick_actions = [
            ("Query Database", self.query_database),
            ("Upload New Video", self.upload_new_video),
            ("Learn More", self.show_help)
        ]
        self.show_quick_buttons(quick_actions)
        
    def query_database(self):
        self.create_message_bubble("Query Database", is_user=True)
        self.hide_quick_buttons()
        
        db_msg = "Please select a video from the database folder"
        
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
            self.create_message_bubble(cancel_msg, is_user=False)
            retry_actions = [
                ("🔄 Try Again", self.query_database),
                ("⬅️ Back to Main Menu", self.restart_conversation)
            ]
            self.root.after(1500, lambda: self.show_quick_buttons(retry_actions))
        
    def handle_video_selection(self, video_file):
        
        #self.create_message_bubble(f"Selected {video_file}", is_user=True)
        video_path = os.path.join(r"C:\Users\adibr\Desktop\video_clip\video_folder", video_file)
        # look for corresponding transcript in transcript folder
        transcript_file = os.path.splitext(video_file)[0] + ".txt"        
        self.transcript_path = os.path.join(r"C:\Users\adibr\Desktop\video_clip\transcripts", transcript_file)
        
        if os.path.exists(self.transcript_path):
            with open(self.transcript_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            
            ready_msg = "Vivi is ready to answer any queries about the video"
            self.create_message_bubble(ready_msg, is_user=False)
            
            self.input_path = video_path
            self.transcript_ready = True
            self.root.after(1500, self.ask_for_query)
            
        else:
            error_msg = "No transcript found for {video_file}"
            self.create_message_bubble(error_msg, is_user=False)
            process_actions = [
                ("Process Video", lambda: self.process_video()),
                ("Back to List", lambda: self.query_database)
            ]
            self.root.after(1500, lambda: self.show_quick_buttons(process_actions))
        
    def upload_new_video(self):
        self.create_message_bubble("Upload New Video", is_user=True)
        self.hide_quick_buttons()
        self.root.after(1200, self.show_file_selection_options)
        
    def show_file_selection_options(self):
        self.show_file_drop_area()
        
        options_msg =  """📁 **File Selection Options:**
        Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"""
        self.create_message_bubble(options_msg, is_user=False)
        
        self.conversation_step = "awaiting_file"
        browse_actions = [("📂 Browse Files", self.browse_file_dialog)]
        self.root.after(1000, lambda: self.show_quick_buttons(browse_actions))
    
    def ask_for_youtube_url(self):
        pass
    
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
            self.create_message_bubble(msg, is_user=False)
            retry_actions = [
                ("📁 Browse Again", self.quick_local_file),
                ("📹 YouTube Instead", self.quick_youtube)
            ]
            self.show_quick_buttons(retry_actions)
            
    def ask_for_query(self):
        msg = "Please type in your queries"
        self.create_message_bubble(msg, is_user=False)
        self.conversation_step="awaiting_query"
    
    def send_message(self, event=None):
        message = self.message_entry.get().strip()
        
        self.create_message_bubble(message, is_user=True)
        self.message_entry.delete(0,'end')
        self.hide_quick_buttons()
        
        if self.conversation_step == "awaiting_url":
            self.handle_url_input(message)
        elif self.conversation_step == "awaiting_query":
            self.handle_query_input(message)
        elif self.conversation_step == "awaiting_file":
            self.handle_file_path_input(message)
    
    def handle_file_path_input(self, filepath):
        filepath = filepath.strip('\'"')
        if os.path.exists(filepath) and os.path.isfile(filepath):
            self.handle_file_selection(filepath)
        else:
            error_msg = f"Couldn't find a file at {filepath}"
            self.create_message_bubble(error_msg, is_user=False)
    
    def handle_url_input(self, url):
        pass
    
    def handle_query_input(self, query):
        pass
#ChatBot Backbone ########
##########################        

# this function is the backbone of having a conversation with the user

##########################        
##########################
    
    def process_video(self):
#Video Processing in case of new video=========================
#=====================================

# This function is the backbone of video processing. It involves:
# 1. Video Extraction   
# 2. Transcription
# We have these function ready they just need to be carefully integrated here!!

#============================================
#============================================

        pass

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
        self.create_message_bubble(restart_msg, is_user=False)
        
        # Show main options
        quick_actions = [
            ("📹 YouTube Video", self.quick_youtube),
            ("📁 Local File", self.quick_local_file)
        ]
        self.root.after(1500, lambda: self.show_quick_buttons(quick_actions))        
        
    
    def run(self):
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()//2)-(900//2)
        y = (self.root.winfo_screenheight()//2)-(700//2)
        
        self.root.geometry(f"900x700+{x}+{y}")
        self.root.mainloop()
        
def main():
    app = VideoSnipperChatbot()
    app.run()
            
if __name__ == "__main__":
    main()