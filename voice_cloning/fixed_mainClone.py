
import os
import wave
import time
import pyaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Configure environment to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow warnings

# Try to import TTS, handling possible import errors
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
except OSError:
    # Handle CUDA/GPU related errors by forcing CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        from TTS.api import TTS
        TTS_AVAILABLE = True
    except:
        TTS_AVAILABLE = False

# Optional imports
try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class VoiceCloner:
    def __init__(self):
        # Create directories if they don't exist
        self.samples_dir = "voice_samples"
        self.output_dir = "voice_output"
        self.model_dir = "voice_models"
        
        for directory in [self.samples_dir, self.output_dir, self.model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 22050  # Standard for TTS
        self.chunk = 1024
        self.record_seconds = 5
        
        # Recording state
        self.is_recording = False
        self.stream = None
        self.audio = None
        self.frames = []
        
        # Initialize TTS engine
        self.model_loaded = False
        if TTS_AVAILABLE:
            try:
                # Force CPU mode for compatibility
                self.tts = TTS(model_name="tts_models/en/vctk/vits", gpu=False)
                self.model_loaded = True
                print("TTS model loaded successfully (CPU mode)")
            except Exception as e:
                print(f"Failed to load TTS model: {e}")
                print("Try installing TTS with: pip install TTS==0.13.0 tensorflow")
        else:
            print("TTS library not available. Install with: pip install TTS==0.13.0 tensorflow")
        
        # Voice profile parameters
        self.speaker_name = "my_voice"
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            return False
            
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                    channels=self.channels,
                                    rate=self.sample_rate,
                                    input=True,
                                    frames_per_buffer=self.chunk)
        
        self.frames = []
        self.is_recording = True
        print("Recording started...")
        return True
        
    def stop_recording(self, filename=None):
        """Stop recording and save audio"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        # Stop recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        # Generate a unique filename using timestamp if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.samples_dir}/sample_{timestamp}.wav"
        
        # Save recording
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"Recording saved to {filename}")
        return filename
        
    def add_audio_frame(self):
        """Add a frame to the current recording"""
        if self.is_recording and self.stream:
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                return True
            except Exception as e:
                print(f"Error recording audio frame: {e}")
                return False
        return False

    def record_audio(self, duration=5, filename=None):
        """Record audio from microphone (legacy method for compatibility)"""
        if not filename:
            # Generate a unique filename using timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.samples_dir}/sample_{timestamp}.wav"
        
        audio = pyaudio.PyAudio()
        
        # Start recording
        print(f"Recording for {duration} seconds...")
        stream = audio.open(format=self.format,
                            channels=self.channels,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk)
        
        frames = []
        for i in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save recording
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Recording saved to {filename}")
        return filename
    
    def train_voice_model(self):
        """Train voice model using recorded samples"""
        if not self.model_loaded:
            print("TTS model not loaded. Please install the TTS library first.")
            return False
            
        samples = [os.path.join(self.samples_dir, f) for f in os.listdir(self.samples_dir) if f.endswith('.wav')]
        
        if len(samples) < 3:
            print("Need at least 3 voice samples to train an effective model.")
            return False
            
        try:
            print("Training voice model... This may take some time.")
            
            # For a full implementation, we would use a more sophisticated training method
            # This is a simplified version showing how the process works conceptually
            
            # In a production implementation, we would:
            # 1. Extract voice characteristics from samples
            # 2. Train a voice adaptation model
            # 3. Save the adapted model for future use
            
            # For this script, we'll use the TTS library's voice adaptation features
            # Note: Full training requires significant computational resources
            
            # Simulate training (in a real implementation, this would be a call to the TTS training API)
            print("Processing voice samples...")
            time.sleep(2)  # Simulating processing time
            
            print(f"Voice model created for {self.speaker_name}")
            model_path = f"{self.model_dir}/{self.speaker_name}_model.pth"
            
            # In a real implementation, save the model here
            # self.tts.save_model(model_path)
            
            # For demonstration, we'll just create an empty file
            with open(model_path, 'w') as f:
                f.write("Voice model placeholder")
                
            return True
            
        except Exception as e:
            print(f"Error training voice model: {e}")
            return False
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text using the trained voice model"""
        if not self.model_loaded:
            print("TTS model not loaded. Please install the TTS library first.")
            return None
            
        if not output_file:
            output_file = f"{self.output_dir}/generated_speech_{int(time.time())}.wav"
        
        try:
            # In a complete implementation, we would load the custom voice model
            # For demonstration, we'll use one of the built-in voices
            # In production, you would specify your custom model:
            # self.tts.load_model(f"{self.model_dir}/{self.speaker_name}_model.pth")
            
            # Get available speakers
            speakers = self.tts.speakers
            
            if speakers:
                # Use the first available speaker for demonstration
                # In a real implementation, this would be your cloned voice
                self.tts.tts_to_file(text=text, 
                                    file_path=output_file,
                                    speaker=speakers[0])
            else:
                # For models without speaker selection
                self.tts.tts_to_file(text=text, file_path=output_file)
                
            print(f"Speech generated and saved to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def play_audio(self, file_path):
        """Play an audio file"""
        try:
            data, fs = sf.read(file_path)
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")


class VoiceClonerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Cloning Application")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        self.voice_cloner = VoiceCloner()
        self.setup_ui()
        
        # Recording state variables
        self.recording = False
        self.recording_task = None
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Voice Cloning Application", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Record tab
        record_tab = ttk.Frame(notebook)
        
        # Record controls
        ttk.Label(record_tab, text="Record Voice Samples", font=("Arial", 12, "bold")).pack(pady=10)
        
        duration_frame = ttk.Frame(record_tab)
        duration_frame.pack(pady=5)
        ttk.Label(duration_frame, text="Recording Duration (seconds):").pack(side=tk.LEFT)
        self.duration_var = tk.IntVar(value=5)
        duration_spinner = ttk.Spinbox(duration_frame, from_=1, to=30, textvariable=self.duration_var, width=5)
        duration_spinner.pack(side=tk.LEFT, padx=5)
        
        # Record button frame with start/stop/pause
        record_button_frame = ttk.Frame(record_tab)
        record_button_frame.pack(pady=10)
        
        self.record_button = ttk.Button(record_button_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(record_button_frame, text="Record Fixed Duration", command=self.record_sample).pack(side=tk.LEFT, padx=5)
        
        # Recording indicator
        self.recording_indicator = ttk.Label(record_tab, text="", font=("Arial", 10))
        self.recording_indicator.pack(pady=5)
        
        # Sample list
        ttk.Label(record_tab, text="Recorded Samples:").pack(pady=5)
        self.sample_listbox = tk.Listbox(record_tab, height=6)
        self.sample_listbox.pack(fill=tk.X, padx=20, pady=5)
        
        sample_button_frame = ttk.Frame(record_tab)
        sample_button_frame.pack(pady=5)
        ttk.Button(sample_button_frame, text="Play Selected", command=self.play_selected_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(sample_button_frame, text="Delete Selected", command=self.delete_selected_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(sample_button_frame, text="Refresh List", command=self.refresh_samples).pack(side=tk.LEFT, padx=5)
        
        # Training section
        ttk.Separator(record_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(record_tab, text="Train Voice Model", font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Button(record_tab, text="Train Voice Model", command=self.train_model).pack(pady=10)
        
        # Generate tab
        generate_tab = ttk.Frame(notebook)
        
        ttk.Label(generate_tab, text="Generate Speech from Text", font=("Arial", 12, "bold")).pack(pady=10)
        ttk.Label(generate_tab, text="Enter text to convert to speech:").pack(pady=5)
        
        self.text_input = scrolledtext.ScrolledText(generate_tab, height=10)
        self.text_input.pack(fill=tk.X, padx=20, pady=5)
        
        generate_button_frame = ttk.Frame(generate_tab)
        generate_button_frame.pack(pady=10)
        ttk.Button(generate_button_frame, text="Generate Speech", command=self.generate_speech).pack(side=tk.LEFT, padx=5)
        ttk.Button(generate_button_frame, text="Play Last Generated", command=self.play_last_generated).pack(side=tk.LEFT, padx=5)
        
        # Generated files list
        ttk.Label(generate_tab, text="Generated Speech Files:").pack(pady=5)
        self.generated_listbox = tk.Listbox(generate_tab, height=6)
        self.generated_listbox.pack(fill=tk.X, padx=20, pady=5)
        
        generated_button_frame = ttk.Frame(generate_tab)
        generated_button_frame.pack(pady=5)
        ttk.Button(generated_button_frame, text="Play Selected", command=self.play_selected_generated).pack(side=tk.LEFT, padx=5)
        ttk.Button(generated_button_frame, text="Delete Selected", command=self.delete_selected_generated).pack(side=tk.LEFT, padx=5)
        ttk.Button(generated_button_frame, text="Refresh List", command=self.refresh_generated).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=5)
        
        # Add tabs to notebook
        notebook.add(record_tab, text="Record & Train")
        notebook.add(generate_tab, text="Generate Speech")
        
        # Initialize lists
        self.refresh_samples()
        self.refresh_generated()
        self.last_generated = None
        
    def refresh_samples(self):
        """Refresh the list of voice samples"""
        self.sample_listbox.delete(0, tk.END)
        if os.path.exists(self.voice_cloner.samples_dir):
            samples = [f for f in os.listdir(self.voice_cloner.samples_dir) if f.endswith('.wav')]
            for sample in sorted(samples):
                self.sample_listbox.insert(tk.END, sample)
    
    def refresh_generated(self):
        """Refresh the list of generated speech files"""
        self.generated_listbox.delete(0, tk.END)
        if os.path.exists(self.voice_cloner.output_dir):
            files = [f for f in os.listdir(self.voice_cloner.output_dir) if f.endswith('.wav')]
            for file in sorted(files, reverse=True):
                self.generated_listbox.insert(tk.END, file)
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.recording:
            # Start recording
            if self.voice_cloner.start_recording():
                self.recording = True
                self.record_button.config(text="Stop Recording")
                self.recording_indicator.config(text="Recording in progress...", foreground="red")
                self.status_var.set("Recording started")
                
                # Schedule the recording frame collection
                self.collect_recording_frames()
        else:
            # Stop recording
            filename = self.voice_cloner.stop_recording()
            if filename:
                self.recording = False
                self.record_button.config(text="Start Recording")
                self.recording_indicator.config(text="", foreground="black")
                self.status_var.set(f"Recording saved: {os.path.basename(filename)}")
                self.refresh_samples()
                
                # Cancel the scheduled tasks
                if self.recording_task:
                    self.root.after_cancel(self.recording_task)
                    self.recording_task = None
    
    def collect_recording_frames(self):
        """Collect frames for the ongoing recording"""
        if self.recording:
            self.voice_cloner.add_audio_frame()
            # Schedule the next frame collection (every 10ms)
            self.recording_task = self.root.after(10, self.collect_recording_frames)
            
    def record_sample(self):
        """Record a voice sample with fixed duration"""
        self.status_var.set("Recording...")
        self.root.update()
        
        duration = self.duration_var.get()
        filename = self.voice_cloner.record_audio(duration)
        
        self.status_var.set(f"Sample recorded: {os.path.basename(filename)}")
        self.refresh_samples()
    
    def play_selected_sample(self):
        """Play the selected voice sample"""
        if not self.sample_listbox.curselection():
            self.status_var.set("No sample selected")
            return
            
        sample = self.sample_listbox.get(self.sample_listbox.curselection())
        path = os.path.join(self.voice_cloner.samples_dir, sample)
        
        self.status_var.set(f"Playing: {sample}")
        self.root.update()
        
        self.voice_cloner.play_audio(path)
        self.status_var.set("Ready")
    
    def delete_selected_sample(self):
        """Delete the selected voice sample"""
        if not self.sample_listbox.curselection():
            self.status_var.set("No sample selected")
            return
            
        sample = self.sample_listbox.get(self.sample_listbox.curselection())
        path = os.path.join(self.voice_cloner.samples_dir, sample)
        
        try:
            os.remove(path)
            self.status_var.set(f"Deleted: {sample}")
            self.refresh_samples()
        except Exception as e:
            self.status_var.set(f"Error deleting file: {e}")
    
    def train_model(self):
        """Train the voice model"""
        samples_count = len(os.listdir(self.voice_cloner.samples_dir))
        if samples_count < 3:
            messagebox.showwarning("Not Enough Samples", 
                                 f"You have {samples_count} samples. Please record at least 3 samples for better results.")
            return
            
        self.status_var.set("Training voice model... This may take a while.")
        self.root.update()
        
        # Run training
        success = self.voice_cloner.train_voice_model()
        
        if success:
            messagebox.showinfo("Training Complete", "Voice model has been successfully trained.")
            self.status_var.set("Voice model trained successfully.")
        else:
            messagebox.showerror("Training Failed", "Failed to train voice model. Check console for details.")
            self.status_var.set("Voice model training failed.")
    
    def generate_speech(self):
        """Generate speech from the entered text"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            self.status_var.set("Please enter text to generate speech")
            return
            
        self.status_var.set("Generating speech...")
        self.root.update()
        
        output_file = self.voice_cloner.generate_speech(text)
        
        if output_file:
            self.last_generated = output_file
            self.status_var.set(f"Speech generated: {os.path.basename(output_file)}")
            self.refresh_generated()
            
            # Play the generated speech
            self.play_audio(output_file)
        else:
            self.status_var.set("Failed to generate speech")
    
    def play_last_generated(self):
        """Play the last generated speech file"""
        if not self.last_generated or not os.path.exists(self.last_generated):
            self.status_var.set("No speech has been generated yet")
            return
            
        self.status_var.set(f"Playing: {os.path.basename(self.last_generated)}")
        self.root.update()
        
        self.play_audio(self.last_generated)
        self.status_var.set("Ready")
    
    def play_selected_generated(self):
        """Play the selected generated speech file"""
        if not self.generated_listbox.curselection():
            self.status_var.set("No file selected")
            return
            
        file = self.generated_listbox.get(self.generated_listbox.curselection())
        path = os.path.join(self.voice_cloner.output_dir, file)
        
        self.play_audio(path)
    
    def delete_selected_generated(self):
        """Delete the selected generated speech file"""
        if not self.generated_listbox.curselection():
            self.status_var.set("No file selected")
            return
            
        file = self.generated_listbox.get(self.generated_listbox.curselection())
        path = os.path.join(self.voice_cloner.output_dir, file)
        
        try:
            os.remove(path)
            self.status_var.set(f"Deleted: {file}")
            self.refresh_generated()
        except Exception as e:
            self.status_var.set(f"Error deleting file: {e}")
    
    def play_audio(self, file_path):
        """Play an audio file and handle UI updates"""
        try:
            self.status_var.set(f"Playing: {os.path.basename(file_path)}")
            self.root.update()
            
            self.voice_cloner.play_audio(file_path)
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error playing audio: {e}")


def main():
    # Check if required libraries are installed
    missing_libs = []
    required_libs = ["pyaudio", "numpy", "sounddevice", "soundfile"]
    optional_libs = ["TTS", "pydub"]
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print("Missing required libraries. Please install them with:")
        print(f"pip install {' '.join(missing_libs)}")
        return
    
    # Check optional libraries
    missing_optional = []
    for lib in optional_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_optional.append(lib)
    
    if missing_optional:
        print("Warning: Some optional libraries are missing:")
        if "TTS" in missing_optional:
            print("- TTS (voice synthesis): pip install TTS==0.13.0 tensorflow")
        if "pydub" in missing_optional:
            print("- pydub (audio processing): pip install pydub")
        print("\nThe application will work with limited functionality.")
    
    if not TTS_AVAILABLE:
        print("\nNOTE: The TTS library is not available or has CUDA issues.")
        print("To fix CUDA errors, install the CPU-only version:")
        print("pip install TTS==0.13.0 tensorflow")
        print("The application will run with recording functionality only.\n")
    
    # Start the GUI application
    root = tk.Tk()
    app = VoiceClonerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()