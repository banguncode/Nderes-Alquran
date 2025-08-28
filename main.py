import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
import torch
import whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
)
import librosa
from datetime import timedelta
import re
import subprocess
from difflib import SequenceMatcher
import wave
import pyaudio
import webbrowser


class AudioRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.frames = []
        self.recording = False
        self.p = None
        self.stream = None
    
    def start_recording(self):
        """Start recording audio"""
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.frames = []
            self.recording = True
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def record_chunk(self):
        """Record a chunk of audio"""
        if self.recording and self.stream:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                return True
            except Exception as e:
                print(f"Error recording chunk: {e}")
                return False
        return False
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self.recording = False
        return self.frames
    
    def save_recording(self, filename):
        """Save recorded audio to file"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False


class QuranSRTGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Nderes Al-Quran")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        # Model setup
        self.model = None
        self.processor = None
        self.whisper_model = None
        self.ffmpeg_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Audio recorder
        self.recorder = AudioRecorder()
        self.recording = False
        self.record_thread = None

        # Model directory setup
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.script_dir, "models", "tarteel-whisper")
        self.whisper_model_dir = os.path.join(self.script_dir, "models", "whisper")

        # Quran dataset
        self.quran_data = None
        self.data_dir = os.path.join(self.script_dir, "data")
        self.quran_json_path = os.path.join(self.data_dir, "quran.json")

        # Variables
        self.audio_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd())

        self.setup_ui()
        self.load_quran_data()
        self.load_model()

    def load_quran_data(self):
        """Load Quran dataset from JSON file"""
        try:
            if os.path.exists(self.quran_json_path):
                with open(self.quran_json_path, "r", encoding="utf-8") as f:
                    self.quran_data = json.load(f)
                self.log_message(
                    f"Dataset Quran berhasil dimuat dari: {self.quran_json_path}"
                )
                self.log_message(f"Jumlah surah: {len(self.quran_data)} surah")
            else:
                self.log_message(
                    f"Dataset Quran tidak ditemukan di: {self.quran_json_path}"
                )
                messagebox.showwarning(
                    "Dataset Missing",
                    f"File quran.json tidak ditemukan di:\n{self.quran_json_path}\n\n"
                    "Aplikasi akan menggunakan deteksi otomatis untuk pemisahan ayat.",
                )
        except Exception as e:
            self.log_message(f"Error memuat dataset Quran: {str(e)}")
            messagebox.showerror("Error", f"Gagal memuat dataset Quran: {str(e)}")

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#34495e", height=100)
        header_frame.pack(fill="x", padx=10, pady=10)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Nderes Alquran",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#34495e",
        )
        title_label.pack(expand=True)

        subtitle_label = tk.Label(
            header_frame,
            text="Generator Subtitle Alquran Menggunakan Model Tarteel AI Whisper",
            font=("Arial", 10),
            fg="#bdc3c7",
            bg="#34495e",
        )
        subtitle_label.pack()

        # Donation frame
        donation_frame = tk.Frame(header_frame, bg="#34495e")
        donation_frame.pack(pady=5)

        donation_label = tk.Label(
            donation_frame,
            text="Dukung pengembangan aplikasi ini:",
            font=("Arial", 9),
            fg="#ecf0f1",
            bg="#34495e",
        )
        donation_label.pack(side="left", padx=5)

        # PayPal button
        paypal_btn = tk.Button(
            donation_frame,
            text="PayPal",
            command=lambda: self.open_donation_link("https://paypal.me/rumhaidar"),
            bg="#3498db",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            padx=10,
            pady=2,
        )
        paypal_btn.pack(side="left", padx=2)

        # Ko-fi button
        kofi_btn = tk.Button(
            donation_frame,
            text="☕ trakteer",
            command=lambda: self.open_donation_link("https://trakteer.id/rumhaidar"),
            bg="#ff5722",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            padx=10,
            pady=2,
        )
        kofi_btn.pack(side="left", padx=2)

        # GitHub button
        github_btn = tk.Button(
            donation_frame,
            text="GitHub",
            command=lambda: self.open_donation_link("https://github.com/banguncode"),
            bg="#2c3e50",
            fg="white",
            font=("Arial", 8, "bold"),
            relief="flat",
            padx=10,
            pady=2,
        )
        github_btn.pack(side="left", padx=2)

        # Main content frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill="both", expand=True, padx=10)

        # Audio input frame
        input_frame = tk.LabelFrame(
            main_frame,
            text="Input Audio",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2c3e50",
        )
        input_frame.pack(fill="x", pady=10)

        # File selection section
        file_section = tk.Frame(input_frame, bg="#2c3e50")
        file_section.pack(fill="x", pady=5)

        tk.Label(
            file_section, text="File Audio:", font=("Arial", 10), fg="white", bg="#2c3e50"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        tk.Entry(
            file_section, textvariable=self.audio_file, width=40, font=("Arial", 10)
        ).grid(row=0, column=1, padx=10, pady=5)

        tk.Button(
            file_section,
            text="Browse",
            command=self.browse_audio_file,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=2, padx=5, pady=5)

        # Separator
        separator = tk.Frame(input_frame, height=2, bg="#7f8c8d")
        separator.pack(fill="x", padx=10, pady=10)

        # Recording section
        record_section = tk.Frame(input_frame, bg="#2c3e50")
        record_section.pack(fill="x", pady=5)

        tk.Label(
            record_section,
            text="Atau Rekam Audio:",
            font=("Arial", 10, "bold"),
            fg="white",
            bg="#2c3e50",
        ).pack(anchor="w", padx=10)

        record_controls = tk.Frame(record_section, bg="#2c3e50")
        record_controls.pack(fill="x", padx=10, pady=5)

        self.record_btn = tk.Button(
            record_controls,
            text="🎤 Mulai Rekam",
            command=self.toggle_recording,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            width=15,
        )
        self.record_btn.pack(side="left", padx=5)

        self.record_status_label = tk.Label(
            record_controls,
            text="Siap merekam",
            font=("Arial", 10),
            fg="#95a5a6",
            bg="#2c3e50",
        )
        self.record_status_label.pack(side="left", padx=10)

        # Recording info
        record_info = tk.Label(
            record_section,
            text="💡 Tips: Pastikan mikrofon berfungsi dan lingkungan tenang untuk hasil terbaik",
            font=("Arial", 9),
            fg="#95a5a6",
            bg="#2c3e50",
        )
        record_info.pack(anchor="w", padx=10, pady=2)

        # Output directory frame
        output_frame = tk.LabelFrame(
            main_frame,
            text="Direktori Output",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2c3e50",
        )
        output_frame.pack(fill="x", pady=10)

        tk.Label(
            output_frame,
            text="Direktori:",
            font=("Arial", 10),
            fg="white",
            bg="#2c3e50",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        tk.Entry(
            output_frame, textvariable=self.output_dir, width=50, font=("Arial", 10)
        ).grid(row=0, column=1, padx=10, pady=5)

        tk.Button(
            output_frame,
            text="Browse",
            command=self.browse_output_dir,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=2, padx=10, pady=5)

        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg="#2c3e50")
        button_frame.pack(fill="x", pady=20)

        self.generate_btn = tk.Button(
            button_frame,
            text="Generate Subtitle SRT",
            command=self.start_generation,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=20,
        )
        self.generate_btn.pack(side="left", padx=10)

        self.clear_btn = tk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=15,
        )
        self.clear_btn.pack(side="left", padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=10)

        # Status and device info
        status_frame = tk.Frame(main_frame, bg="#2c3e50")
        status_frame.pack(fill="x", pady=5)

        self.device_label = tk.Label(
            status_frame,
            text=f"Device: {self.device.upper()}",
            font=("Arial", 10, "bold"),
            fg="#f39c12",
            bg="#2c3e50",
        )
        self.device_label.pack(side="left")

        self.status_label = tk.Label(
            status_frame,
            text="Status: Siap",
            font=("Arial", 10),
            fg="#2ecc71",
            bg="#2c3e50",
        )
        self.status_label.pack(side="right")

        # Log text area
        log_frame = tk.LabelFrame(
            main_frame,
            text="Log Proses",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2c3e50",
        )
        log_frame.pack(fill="both", expand=True, pady=10)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#34495e", fg="white", font=("Consolas", 9)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def open_donation_link(self, url):
        """Open donation link in browser"""
        try:
            webbrowser.open(url)
            self.log_message(f"Membuka link donasi: {url}")
        except Exception as e:
            self.log_message(f"Error membuka link: {str(e)}")
            messagebox.showerror("Error", f"Gagal membuka link: {str(e)}")

    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start audio recording"""
        try:
            # Check if pyaudio is available
            try:
                import pyaudio
            except ImportError:
                messagebox.showerror(
                    "Error", 
                    "PyAudio tidak ditemukan!\n\n"
                    "Install dengan: pip install pyaudio\n"
                    "Atau download dari: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"
                )
                return

            if self.recorder.start_recording():
                self.recording = True
                self.record_btn.config(text="⏹️ Stop Rekam", bg="#c0392b")
                self.record_status_label.config(text="🔴 Merekam...", fg="#e74c3c")
                self.log_message("Mulai merekam audio...")
                
                # Start recording thread
                self.record_thread = threading.Thread(target=self.recording_loop, daemon=True)
                self.record_thread.start()
            else:
                messagebox.showerror("Error", "Gagal memulai perekaman audio!")
                
        except Exception as e:
            self.log_message(f"Error starting recording: {str(e)}")
            messagebox.showerror("Error", f"Gagal memulai perekaman: {str(e)}")

    def recording_loop(self):
        """Recording loop in separate thread"""
        while self.recording:
            if not self.recorder.record_chunk():
                break
            # Small delay to prevent high CPU usage
            threading.Event().wait(0.01)

    def stop_recording(self):
        """Stop audio recording and save file"""
        try:
            self.recording = False
            self.recorder.stop_recording()
            
            self.record_btn.config(text="🎤 Mulai Rekam", bg="#e74c3c")
            self.record_status_label.config(text="💾 Menyimpan...", fg="#f39c12")
            
            # Save recording
            timestamp = threading.current_thread().ident
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(self.output_dir.get(), filename)
            
            if self.recorder.save_recording(filepath):
                self.audio_file.set(filepath)
                self.record_status_label.config(text="✅ Rekaman tersimpan", fg="#27ae60")
                self.log_message(f"Rekaman audio tersimpan: {filepath}")
                
                # Auto-generate after 2 seconds
                self.root.after(2000, lambda: self.record_status_label.config(text="Siap merekam", fg="#95a5a6"))
                
                # Ask if user wants to generate subtitle immediately
                if messagebox.askyesno("Rekaman Selesai", "Rekaman audio berhasil disimpan!\n\nMulai generate subtitle sekarang?"):
                    self.start_generation()
            else:
                self.record_status_label.config(text="❌ Gagal menyimpan", fg="#e74c3c")
                messagebox.showerror("Error", "Gagal menyimpan rekaman audio!")
                
        except Exception as e:
            self.log_message(f"Error stopping recording: {str(e)}")
            self.record_status_label.config(text="❌ Error", fg="#e74c3c")
            messagebox.showerror("Error", f"Gagal menghentikan perekaman: {str(e)}")

    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear log text"""
        self.log_text.delete(1.0, tk.END)

    def browse_audio_file(self):
        """Browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Pilih File Audio",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        if filename:
            self.audio_file.set(filename)

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Pilih Direktori Output")
        if directory:
            self.output_dir.set(directory)

    def create_model_directories(self):
        """Create model directories if they don't exist"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.whisper_model_dir, exist_ok=True)

    def check_ffmpeg(self):
        """Check if ffmpeg is available"""
        # First check local directory
        local_ffmpeg = os.path.join(self.script_dir, "ffmpeg.exe")
        if os.path.exists(local_ffmpeg):
            self.ffmpeg_path = local_ffmpeg
            self.log_message(f"FFmpeg ditemukan di direktori lokal: {local_ffmpeg}")
            return True

        # Check ffmpeg folder in same directory
        ffmpeg_folder = os.path.join(self.script_dir, "ffmpeg", "bin", "ffmpeg.exe")
        if os.path.exists(ffmpeg_folder):
            self.ffmpeg_path = ffmpeg_folder
            self.log_message(f"FFmpeg ditemukan di folder ffmpeg: {ffmpeg_folder}")
            return True

        # Check system PATH
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.ffmpeg_path = "ffmpeg"
                self.log_message("FFmpeg ditemukan di system PATH")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return False

    def install_ffmpeg_instructions(self):
        """Show instructions for installing ffmpeg"""
        msg = f"""FFmpeg tidak ditemukan di sistem Anda.

Aplikasi mencari FFmpeg di lokasi berikut:
1. {os.path.join(self.script_dir, "ffmpeg.exe")}
2. {os.path.join(self.script_dir, "ffmpeg", "bin", "ffmpeg.exe")}
3. System PATH

Untuk menggunakan aplikasi ini:

OPSI 1 - Copy FFmpeg ke direktori script:
- Download FFmpeg dari: https://ffmpeg.org/download.html
- Extract dan copy ffmpeg.exe ke direktori yang sama dengan script ini

OPSI 2 - Buat folder ffmpeg:
- Buat folder 'ffmpeg' di direktori script
- Buat subfolder 'bin'
- Copy ffmpeg.exe ke dalam folder bin

OPSI 3 - Install ke system:
- Install menggunakan chocolatey: choco install ffmpeg
- Atau menggunakan scoop: scoop install ffmpeg
- Atau tambahkan FFmpeg ke system PATH

Restart aplikasi setelah mengcopy/install FFmpeg."""

        messagebox.showwarning("FFmpeg Required", msg)

    def load_model(self):
        """Load the Whisper model"""

        def load():
            try:
                # Check for ffmpeg first
                if not self.check_ffmpeg():
                    self.log_message(
                        "Warning: FFmpeg tidak ditemukan. Ini diperlukan untuk memproses audio."
                    )
                    self.status_label.config(text="Status: FFmpeg Required")
                    self.install_ffmpeg_instructions()
                    return

                self.create_model_directories()

                self.log_message("Memeriksa model Tarteel AI Whisper...")
                self.status_label.config(text="Status: Memeriksa Model")

                # Check if model exists locally
                if os.path.exists(os.path.join(self.model_dir, "config.json")):
                    self.log_message(
                        "Model ditemukan secara lokal, memuat dari direktori..."
                    )

                    self.processor = WhisperProcessor.from_pretrained(self.model_dir)
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_dir
                    )
                else:
                    self.log_message(
                        "Model tidak ditemukan, mengunduh dari Hugging Face..."
                    )
                    self.status_label.config(text="Status: Mengunduh Model")

                    # Download and save model locally
                    self.processor = WhisperProcessor.from_pretrained(
                        "tarteel-ai/whisper-base-ar-quran", cache_dir=self.model_dir
                    )
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        "tarteel-ai/whisper-base-ar-quran", cache_dir=self.model_dir
                    )

                    # Save model locally for future use
                    self.log_message("Menyimpan model secara lokal...")
                    self.processor.save_pretrained(self.model_dir)
                    self.model.save_pretrained(self.model_dir)

                # Set generation config for timestamp support
                try:
                    generation_config = GenerationConfig.from_pretrained(
                        "openai/whisper-base"
                    )
                    self.model.generation_config = generation_config
                except:
                    self.log_message(
                        "Warning: Tidak dapat memuat generation config, menggunakan default"
                    )

                self.model.to(self.device)

                # Load Whisper model for timestamps
                self.log_message("Memuat Whisper model untuk timestamp...")
                self.load_whisper_model()

                self.log_message(f"Model berhasil dimuat pada {self.device}")
                self.log_message(f"Model disimpan di: {self.model_dir}")
                self.status_label.config(text="Status: Model Siap")

            except Exception as e:
                self.log_message(f"Error memuat model: {str(e)}")
                self.status_label.config(text="Status: Error Model")
                messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")

        threading.Thread(target=load, daemon=True).start()

    def setup_ffmpeg_environment(self):
        """Setup FFmpeg environment variables for local usage"""
        if self.ffmpeg_path and self.ffmpeg_path != "ffmpeg":
            # Add FFmpeg directory to PATH for this process only
            ffmpeg_dir = os.path.dirname(self.ffmpeg_path)
            current_path = os.environ.get("PATH", "")

            # Only add if not already in PATH
            if ffmpeg_dir not in current_path:
                # Prepend to PATH so it takes priority
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
                self.log_message(
                    f"FFmpeg directory ditambahkan ke PATH sementara: {ffmpeg_dir}"
                )

            # Also set FFMPEG_BINARY for direct usage
            os.environ["FFMPEG_BINARY"] = self.ffmpeg_path

            # Set additional environment variables that some libraries check
            os.environ["FFMPEG_EXECUTABLE"] = self.ffmpeg_path

        return self.ffmpeg_path if self.ffmpeg_path else None

    def load_whisper_model(self):
        """Load Whisper model from local directory or download if needed"""
        try:
            # Setup FFmpeg environment first
            ffmpeg_binary = self.setup_ffmpeg_environment()

            # Set whisper cache directory
            os.environ["WHISPER_CACHE_DIR"] = self.whisper_model_dir

            # Patch whisper's audio loading to use our local ffmpeg
            if ffmpeg_binary and ffmpeg_binary != "ffmpeg":
                self.patch_whisper_ffmpeg(ffmpeg_binary)

            # Try to load model with explicit download root
            self.log_message("Memuat model Whisper base...")
            self.whisper_model = whisper.load_model(
                "base", download_root=self.whisper_model_dir
            )
            self.log_message("Whisper model berhasil dimuat")

        except Exception as e:
            self.log_message(f"Error loading Whisper model: {str(e)}")
            self.log_message("Mencoba memuat model dengan metode alternatif...")
            try:
                # Fallback method
                self.whisper_model = whisper.load_model("base")
                self.log_message("Whisper model berhasil dimuat (metode fallback)")
            except Exception as e2:
                self.log_message(f"Error loading Whisper fallback: {str(e2)}")
                raise e2

    def patch_whisper_ffmpeg(self, ffmpeg_path):
        """Patch whisper to use local ffmpeg"""
        try:
            import whisper.audio

            # Store original function
            if not hasattr(self, "_original_load_audio"):
                self._original_load_audio = whisper.audio.load_audio

            # Create patched function
            def patched_load_audio(file: str, sr: int = 16000):
                """Load audio using local ffmpeg"""
                import subprocess
                import numpy as np

                try:
                    # Use our local ffmpeg
                    cmd = [
                        ffmpeg_path,
                        "-nostdin",
                        "-threads",
                        "0",
                        "-i",
                        file,
                        "-f",
                        "s16le",
                        "-ac",
                        "1",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        str(sr),
                        "-",
                    ]

                    out = subprocess.run(cmd, capture_output=True, check=True).stdout

                except subprocess.CalledProcessError as e:
                    self.log_message(f"FFmpeg error: {e.stderr.decode()}")
                    # Fallback to original function
                    return self._original_load_audio(file, sr)

                return (
                    np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
                )

            # Apply patch
            whisper.audio.load_audio = patched_load_audio
            self.log_message(
                f"Whisper di-patch untuk menggunakan FFmpeg lokal: {ffmpeg_path}"
            )

        except Exception as e:
            self.log_message(f"Warning: Gagal mem-patch whisper: {str(e)}")
            self.log_message("Menggunakan konfigurasi default")

    def format_timestamp(self, seconds):
        """Format seconds to SRT timestamp format"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return (
            f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
        )

    def normalize_arabic_text(self, text):
        """Normalize Arabic text for better matching"""
        # Remove diacritics and normalize
        text = re.sub(
            r"[\u064B-\u0652\u0670\u0640]", "", text
        )  # Remove diacritics and tatweel
        text = re.sub(r"[۝﴿﴾]", "", text)  # Remove verse markers
        text = re.sub(r"[٠-٩]+", "", text)  # Remove Arabic numerals
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
        return text

    def find_best_surah_match(self, transcription):
        """Find the best matching surah from the dataset"""
        if not self.quran_data:
            return None, None

        normalized_transcription = self.normalize_arabic_text(transcription)
        best_match = None
        best_score = 0

        self.log_message("Mencari surah yang cocok dengan hasil transkripsi...")

        for surah_num, surah_data in self.quran_data.items():
            # Combine all ayahs in the surah
            surah_text = " ".join(surah_data["text"].values())
            normalized_surah = self.normalize_arabic_text(surah_text)

            # Calculate similarity
            similarity = SequenceMatcher(
                None, normalized_transcription, normalized_surah
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = surah_data

        if best_match and best_score > 0.3:  # Threshold for match
            self.log_message(
                f"Surah ditemukan: {best_match['name_latin']} (skor: {best_score:.3f})"
            )
            return best_match, best_score
        else:
            self.log_message("Tidak ada surah yang cocok ditemukan dalam dataset")
            return None, 0

    def align_ayahs_with_segments(self, surah_data, whisper_segments, transcription):
        """Align ayahs from dataset with whisper segments"""
        if not surah_data:
            return self.create_fallback_segments(transcription, whisper_segments)

        ayahs = surah_data["text"]
        ayah_segments = []

        self.log_message(
            f"Menyelaraskan {len(ayahs)} ayat dengan {len(whisper_segments)} segmen audio..."
        )

        # Normalize transcription
        normalized_transcription = self.normalize_arabic_text(transcription)

        # Create segments for each ayah
        total_segments = len(whisper_segments)
        total_ayahs = len(ayahs)

        if total_segments == 0:
            self.log_message("Warning: Tidak ada segmen audio ditemukan")
            return self.create_fallback_segments(transcription, [])

        # Method 1: Direct mapping if segments match ayahs
        if total_segments >= total_ayahs:
            self.log_message("Menggunakan pemetaan langsung segmen ke ayat...")
            segments_per_ayah = total_segments / total_ayahs

            for i, (ayah_num, ayah_text) in enumerate(ayahs.items()):
                # Calculate which whisper segments correspond to this ayah
                start_segment_idx = int(i * segments_per_ayah)
                end_segment_idx = min(
                    int((i + 1) * segments_per_ayah), total_segments - 1
                )

                if start_segment_idx < total_segments:
                    start_time = whisper_segments[start_segment_idx]["start"]
                    end_time = whisper_segments[end_segment_idx]["end"]

                    ayah_segments.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "text": ayah_text,
                            "ayah_number": ayah_num,
                        }
                    )
        else:
            # Method 2: Distribute time evenly among ayahs
            self.log_message("Menggunakan distribusi waktu merata untuk ayat...")
            total_duration = whisper_segments[-1]["end"] if whisper_segments else 60
            ayah_duration = total_duration / total_ayahs

            for i, (ayah_num, ayah_text) in enumerate(ayahs.items()):
                start_time = i * ayah_duration
                end_time = (i + 1) * ayah_duration

                # Try to align with nearest whisper segment
                if i < len(whisper_segments):
                    # Use whisper segment timing as reference
                    whisper_start = whisper_segments[i]["start"]
                    whisper_end = whisper_segments[i]["end"]

                    # Interpolate between calculated and whisper timing
                    start_time = (start_time + whisper_start) / 2
                    end_time = (end_time + whisper_end) / 2

                ayah_segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "text": ayah_text,
                        "ayah_number": ayah_num,
                    }
                )

        # Method 3: Text-based alignment for better accuracy
        self.log_message(
            "Melakukan alignment berbasis teks untuk akurasi yang lebih baik..."
        )
        ayah_segments = self.refine_alignment_with_text_matching(
            ayah_segments, whisper_segments, transcription
        )

        self.log_message(f"Berhasil membuat {len(ayah_segments)} segmen ayat")
        return ayah_segments

    def refine_alignment_with_text_matching(
        self, ayah_segments, whisper_segments, transcription
    ):
        """Refine alignment using text matching"""
        if not whisper_segments or not ayah_segments:
            return ayah_segments

        refined_segments = []

        for ayah_segment in ayah_segments:
            ayah_text = self.normalize_arabic_text(ayah_segment["text"])
            best_match_score = 0
            best_timing = None

            # Find whisper segments that best match this ayah
            for whisper_seg in whisper_segments:
                whisper_text = self.normalize_arabic_text(whisper_seg.get("text", ""))

                if whisper_text:
                    similarity = SequenceMatcher(None, ayah_text, whisper_text).ratio()

                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_timing = {
                            "start": whisper_seg["start"],
                            "end": whisper_seg["end"],
                        }

            # Use best match timing if found, otherwise keep original
            if best_timing and best_match_score > 0.4:
                refined_segment = ayah_segment.copy()
                refined_segment.update(best_timing)
                refined_segments.append(refined_segment)
            else:
                refined_segments.append(ayah_segment)

        return refined_segments

    def create_fallback_segments(self, transcription, whisper_segments):
        """Create fallback segments when no dataset match is found"""
        self.log_message("Menggunakan deteksi otomatis untuk pemisahan ayat...")

        # Try to detect verse patterns in transcription
        verse_patterns = [
            r"۝[٠-٩]+",  # Verse markers with numbers
            r"﴿[٠-٩]+﴾",  # Verse numbers in brackets
            r"[۝﴾][٠-٩]+",  # General verse ending patterns
        ]

        verses = []
        current_verse = ""

        # Split by verse markers
        for pattern in verse_patterns:
            parts = re.split(f"({pattern})", transcription)
            if len(parts) > 1:
                for part in parts:
                    current_verse += part
                    if re.match(pattern, part):
                        if current_verse.strip():
                            verses.append(current_verse.strip())
                            current_verse = ""
                break

        # Add remaining text
        if current_verse.strip():
            verses.append(current_verse.strip())

        # If no verses detected, split by punctuation or length
        if not verses:
            verses = re.split(r"[.،؛]{2,}|\s{5,}", transcription)
            verses = [v.strip() for v in verses if v.strip()]

        if not verses:
            verses = [transcription]  # Fallback to entire transcription

        # Create segments
        segments = []
        total_duration = whisper_segments[-1]["end"] if whisper_segments else 60
        verse_duration = total_duration / len(verses)

        for i, verse_text in enumerate(verses):
            start_time = i * verse_duration
            end_time = (i + 1) * verse_duration

            # Align with whisper segments if available
            if i < len(whisper_segments):
                whisper_start = whisper_segments[i]["start"]
                whisper_end = whisper_segments[i]["end"]
                start_time = (start_time + whisper_start) / 2
                end_time = (end_time + whisper_end) / 2

            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": verse_text,
                    "ayah_number": str(i + 1),
                }
            )

        return segments

    def validate_audio_file(self, audio_path):
        """Validate if audio file exists and is accessible"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

        if not os.path.isfile(audio_path):
            raise ValueError(f"Path bukan file: {audio_path}")

        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("File audio kosong")

        self.log_message(f"File audio valid: {audio_path} ({file_size} bytes)")

    def transcribe_audio_with_segments(self, audio_path):
        """Transcribe audio and return segments with timestamps"""
        try:
            # Validate audio file first
            self.validate_audio_file(audio_path)

            self.log_message("Memuat file audio...")

            if not self.whisper_model:
                raise RuntimeError("Whisper model tidak tersedia")

            # Ensure FFmpeg environment is set up
            ffmpeg_binary = self.setup_ffmpeg_environment()

            self.log_message("Memproses audio dengan Whisper untuk timestamp...")

            # Convert path to raw string to handle Windows paths
            audio_path = os.path.normpath(audio_path)
            self.log_message(f"Path audio yang akan diproses: {audio_path}")

            # Test FFmpeg access before transcription
            if ffmpeg_binary and ffmpeg_binary != "ffmpeg":
                self.test_ffmpeg_access(audio_path, ffmpeg_binary)

            # Transkripsi dengan timestamp per segment
            result = self.whisper_model.transcribe(
                audio_path, language="ar", word_timestamps=False, verbose=True
            )

            if not result or "segments" not in result:
                raise RuntimeError(
                    "Whisper transcription gagal - tidak ada hasil yang diperoleh"
                )

            self.log_message(
                f"Whisper transcription berhasil, {len(result['segments'])} segmen ditemukan"
            )

            # Juga gunakan model Tarteel untuk transkripsi yang lebih akurat
            self.log_message(
                "Memproses dengan model Tarteel untuk akurasi yang lebih baik..."
            )

            try:
                audio, sr = librosa.load(audio_path, sr=16000)
                inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs["input_features"], max_length=448
                    )

                tarteel_transcription = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                self.log_message("Transkripsi Tarteel berhasil")
                self.log_message(f"Hasil transkripsi: {tarteel_transcription[:100]}...")

            except Exception as e:
                self.log_message(f"Warning: Gagal menggunakan model Tarteel: {str(e)}")
                self.log_message("Menggunakan hasil Whisper saja...")
                tarteel_transcription = result["text"]

            self.log_message("Menggabungkan hasil transkripsi...")

            # Gunakan transkripsi Tarteel yang lebih akurat untuk teks Arab
            # Dan timestamp dari Whisper untuk segmentasi
            segments = result["segments"]

            return tarteel_transcription, segments

        except FileNotFoundError as e:
            self.log_message(f"Error: File tidak ditemukan - {str(e)}")
            raise e
        except Exception as e:
            self.log_message(f"Error dalam transkripsi: {str(e)}")
            self.log_message(f"Error type: {type(e).__name__}")
            import traceback

            self.log_message(f"Traceback: {traceback.format_exc()}")
            raise e

    def test_ffmpeg_access(self, audio_path, ffmpeg_binary):
        """Test if FFmpeg can access the audio file"""
        try:
            import subprocess

            # Test with a simple probe command
            cmd = [ffmpeg_binary, "-i", audio_path, "-f", "null", "-t", "1", "-"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.log_message("FFmpeg berhasil mengakses file audio")
            else:
                self.log_message(f"FFmpeg warning: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.log_message("FFmpeg test timeout - file mungkin besar")
        except Exception as e:
            self.log_message(f"FFmpeg test error: {str(e)}")
            # Don't raise - let whisper try anyway

    def generate_srt(self, ayah_segments, output_path):
        """Generate SRT file from ayah segments"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(ayah_segments, 1):
                    start_time = self.format_timestamp(segment["start"])
                    end_time = self.format_timestamp(segment["end"])

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")

            self.log_message(f"File SRT berhasil disimpan: {output_path}")
            self.log_message(f"Total {len(ayah_segments)} ayat dalam subtitle")

        except Exception as e:
            self.log_message(f"Error menulis file SRT: {str(e)}")
            raise e

    def start_generation(self):
        """Start the SRT generation process"""
        if not self.audio_file.get():
            messagebox.showerror("Error", "Pilih file audio terlebih dahulu!")
            return

        if not self.model or not self.whisper_model:
            messagebox.showerror("Error", "Model belum dimuat!")
            return

        def generate():
            try:
                self.generate_btn.config(state="disabled")
                self.progress.start()
                self.status_label.config(text="Status: Memproses...")

                audio_path = self.audio_file.get()
                output_dir = self.output_dir.get()

                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_quran.srt")

                self.log_message(f"Memulai pemrosesan: {audio_path}")

                # Transcribe audio with segments
                transcription, whisper_segments = self.transcribe_audio_with_segments(
                    audio_path
                )

                self.log_message("Mencari surah yang cocok dalam dataset...")

                # Find matching surah in dataset
                matching_surah, match_score = self.find_best_surah_match(transcription)

                # Align ayahs with audio segments
                ayah_segments = self.align_ayahs_with_segments(
                    matching_surah, whisper_segments, transcription
                )

                if matching_surah:
                    self.log_message(
                        f"Menggunakan data surah: {matching_surah['name_latin']}"
                    )
                    self.log_message(f"Akurasi pencocokan: {match_score:.1%}")
                else:
                    self.log_message("Menggunakan deteksi otomatis ayat")

                # Generate SRT file
                self.generate_srt(ayah_segments, output_path)

                self.log_message("Proses selesai!")
                self.status_label.config(text="Status: Selesai")

                messagebox.showinfo(
                    "Sukses", f"File SRT berhasil dibuat!\n{output_path}"
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.log_message(error_msg)
                self.status_label.config(text="Status: Error")
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

            finally:
                self.progress.stop()
                self.generate_btn.config(state="normal")

        threading.Thread(target=generate, daemon=True).start()


def main():
    root = tk.Tk()
    app = QuranSRTGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
