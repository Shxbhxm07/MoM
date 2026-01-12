from io import BytesIO
from typing import Optional, List, Dict
import os
import tempfile
import threading
import time
import queue
from datetime import datetime
import warnings
import json
import asyncio
import uuid
import hashlib
import uvicorn
import traceback
from scipy import signal


import numpy as np
from scipy.io import wavfile

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

# PDF generation
_old_md5 = hashlib.md5
def _md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _old_md5(*args, **kwargs)
hashlib.md5 = _md5

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")

# Try to import sounddevice for live recording
try:
    import sounddevice as sd
    LIVE_RECORDING_AVAILABLE = True
except ImportError:
    LIVE_RECORDING_AVAILABLE = False
    print("[WARN] sounddevice not available, live recording disabled")

# =============================================================================
# CONFIG
# =============================================================================
WHISPER_URL = "http://speech-unified:8000"
LLAMA_URL = "http://llama-summarizer:8001"
NEMO_URL = "http://nemo-diarizer:8003"

TIMEOUT = 600.0
TRANSCRIPTION_TIMEOUT = 30.0  # Increased from 15.0 to handle longer segments

# Audio settings
WHISPER_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCKSIZE = 8192

# Legacy settings (for file upload)
SEGMENT_INTERVAL = 5.0
MIN_AUDIO_LENGTH = 0.5
SILENCE_THRESHOLD = 0.01

# Live recording settings (FASTER)
LIVE_CHUNK_DURATION = 2.0  # Process every 2 seconds
LIVE_MIN_AUDIO_LENGTH = 0.8  # Minimum 0.8s audio
LIVE_SILENCE_THRESHOLD = 0.005  # Lower threshold for better detection
LIVE_VAD_ENABLED = True  # Enable Voice Activity Detection

FILE_CACHE_SIZE_MB = 500

# =============================================================================
# DEVICE DETECTION
# =============================================================================
def detect_supported_sample_rate() -> int:
    if not LIVE_RECORDING_AVAILABLE:
        return 48000
        
    sample_rates_to_try = [48000, 44100, 32000, 24000, 22050, 16000, 8000]
    print("[INFO] Detecting supported audio sample rates...")

    try:
        device_info = sd.query_devices(kind="input")
        default_sr = int(device_info.get("default_samplerate", 48000))
        print(f"[INFO] Default device: {device_info.get('name', 'Unknown')}")
        print(f"[INFO] Default sample rate: {default_sr}")
        if default_sr not in sample_rates_to_try:
            sample_rates_to_try.insert(0, default_sr)
    except Exception as e:
        print(f"[WARN] Could not query device: {e}")

    for sr in sample_rates_to_try:
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype=np.float32, blocksize=1024):
                print(f"[INFO] ✓ Sample rate {sr} Hz is supported")
                return sr
        except Exception:
            continue

    return 48000

RECORDING_SAMPLE_RATE = detect_supported_sample_rate()
INPUT_DEVICE_ID = 4  # Device 4: MH148 USB Audio

# =============================================================================
# Pydantic Models
# =============================================================================
class TranscriptionResponse(BaseModel):
    formatted_text: str
    full_text: str

class LiveStatusResponse(BaseModel):
    is_recording: bool
    duration: float
    word_count: int
    total_segments: int
    latest_text: Optional[str] = None
    transcript: str
    audio_level: float

class ExportRequest(BaseModel):
    content: str
    title: Optional[str] = None

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 300

class CompleteExportRequest(BaseModel):
    formatted_transcript: str
    summary: Optional[str] = None
    raw_transcript: str
    filename: str
    speaker_count: int

# =============================================================================
# PROGRESS TRACKING & CACHING
# =============================================================================
class ProgressTracker:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str):
        with self._lock:
            self.jobs[job_id] = {
                "stage": "queued",
                "progress": 0.0,
                "message": "Job queued",
                "eta_seconds": None,
                "start_time": time.time()
            }

    def update_progress(self, job_id: str, stage: str, progress: float, message: str, eta: Optional[float] = None):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update({
                    "stage": stage,
                    "progress": progress,
                    "message": message,
                    "eta_seconds": eta
                })

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return self.jobs.get(job_id)

class FileCache:
    def __init__(self, max_size_mb: int = 500):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self._lock = threading.Lock()

    def get_file_hash(self, file_content: bytes) -> str:
        return hashlib.sha256(file_content).hexdigest()

    def get(self, file_hash: str) -> Optional[Dict]:
        with self._lock:
            if file_hash in self.cache:
                return self.cache[file_hash]
            return None

    def set(self, file_hash: str, result: Dict, file_size: int):
        with self._lock:
            while self.current_size + file_size > self.max_size and self.cache:
                oldest_key = next(iter(self.cache))
                oldest = self.cache.pop(oldest_key)
                self.current_size -= oldest.get("file_size", 0)

            self.cache[file_hash] = {
                **result,
                "file_size": file_size,
                "cached_at": time.time()
            }
            self.current_size += file_size

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.current_size = 0

progress_tracker = ProgressTracker()
file_cache = FileCache(max_size_mb=FILE_CACHE_SIZE_MB)

# =============================================================================
# UTILS
# =============================================================================
async def wait_for_service(url: str, service_name: str, max_retries: int = 30, delay: int = 5) -> bool:
    """Wait for a service to become available"""
    print(f"[INFO] Waiting for {service_name} to be ready...")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"[INFO] ✓ {service_name} is ready!")
                    return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[INFO] {service_name} not ready yet (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[ERROR] {service_name} failed to start: {e}")
                return False
    
    return False

def calculate_rms(audio_data: np.ndarray) -> float:
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_data))))

def count_words(text: str) -> int:
    return len(text.split()) if text else 0

def find_speaker_for_time(start_time: float, end_time: float, speaker_segments: list) -> str:
    """Find the speaker for a given time range"""
    max_overlap = 0
    best_speaker = "SPEAKER_00"
    
    mid_time = (start_time + end_time) / 2
    
    for spk_seg in speaker_segments:
        spk_start = spk_seg.get("start", 0.0)
        spk_end = spk_seg.get("end", 0.0)
        
        if spk_start <= mid_time <= spk_end:
            return spk_seg.get("speaker", "SPEAKER_00").upper()
        
        overlap_start = max(start_time, spk_start)
        overlap_end = min(end_time, spk_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = spk_seg.get("speaker", "SPEAKER_00")
    
    return best_speaker.upper()


def format_diarized_transcript(transcription_result: dict, diarization_result: dict) -> str:
    """Format transcript with speaker labels"""
    formatted_lines = []
    
    transcription_segments = transcription_result.get("segments", [])
    speaker_segments = diarization_result.get("speakers", [])
    
    if not transcription_segments:
        return transcription_result.get("text", "")
    
    for trans_seg in transcription_segments:
        start_time = trans_seg.get("start", 0.0)
        end_time = trans_seg.get("end", start_time + 1.0)
        text = trans_seg.get("text", "").strip()
        
        if not text:
            continue
        
        words = trans_seg.get("words", [])
        
        if words and len(words) > 0:
            current_speaker = None
            current_text = []
            current_start = None
            
            for word_info in words:
                word = word_info.get("word", "").strip()
                word_start = word_info.get("start", start_time)
                word_end = word_info.get("end", word_start + 0.1)
                
                speaker = find_speaker_for_time(word_start, word_end, speaker_segments)
                
                if speaker != current_speaker and current_text:
                    minutes = int(current_start // 60)
                    seconds = int(current_start % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    formatted_lines.append(
                        f"[{time_str}] {current_speaker}: {' '.join(current_text)}"
                    )
                    current_text = []
                
                if not current_text:
                    current_start = word_start
                    current_speaker = speaker
                
                current_text.append(word)
            
            if current_text:
                minutes = int(current_start // 60)
                seconds = int(current_start % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                formatted_lines.append(
                    f"[{time_str}] {current_speaker}: {' '.join(current_text)}"
                )
        else:
            speaker = find_speaker_for_time(start_time, end_time, speaker_segments)
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            formatted_lines.append(f"[{time_str}] {speaker}: {text}")
    
    return "\n".join(formatted_lines)


def build_pdf_buffer(content: str, title: str = "AI Pipeline Transcription") -> BytesIO:
    buffer = BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setTitle(title)
        width, height = letter
        x, y = 50, height - 50
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, title)
        y -= 24
        
        c.setFont("Helvetica", 9)
        c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 14
        c.drawString(x, y, f"Total Words: {count_words(content)}")
        y -= 24
        
        c.setFont("Helvetica", 10)
        for line in content.split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(x, y, line[:100])  # Limit line length
            y -= 14
        
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"[WARN] PDF generation failed: {e}")
        buffer.seek(0)
        return buffer

# =============================================================================
# LIVE RECORDING STATE (only if sounddevice available)
# =============================================================================
class RecordingState:
    def __init__(self):
        self._should_stop = threading.Event()
        self._is_recording = threading.Event()
        self._audio_queue = queue.Queue(maxsize=1000)
        self._segment_queue = queue.Queue()
        self._audio_level = 0.0
        self._level_lock = threading.Lock()
        self._full_audio: List[np.ndarray] = []

    def reset(self):
        self._should_stop.clear()
        self._is_recording.clear()
        self._audio_level = 0.0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                break
        while not self._segment_queue.empty():
            try:
                self._segment_queue.get_nowait()
            except:
                break
        self._full_audio = []

    def start(self):
        self._should_stop.clear()
        self._is_recording.set()

    def stop(self):
        self._should_stop.set()

    def mark_stopped(self):
        self._is_recording.clear()

    @property
    def should_stop(self) -> bool:
        return self._should_stop.is_set()

    @property
    def is_recording(self) -> bool:
        return self._is_recording.is_set()

    def add_audio(self, audio_chunk: np.ndarray):
        try:
            self._audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio_chunk)
            except:
                pass

    def add_full_audio(self, audio_chunk: np.ndarray):
        self._full_audio.append(audio_chunk.copy())

    def get_full_audio(self) -> Optional[np.ndarray]:
        if not self._full_audio:
            return None
        return np.concatenate(self._full_audio)

    def get_audio_chunks(self, timeout=0.1) -> List[np.ndarray]:
        chunks = []
        try:
            while True:
                chunk = self._audio_queue.get(timeout=timeout if not chunks else 0.001)
                chunks.append(chunk)
        except queue.Empty:
            pass
        return chunks

    def add_segment(self, segment: dict):
        self._segment_queue.put(segment)

    def get_segments(self) -> List[dict]:
        segments = []
        while True:
            try:
                segments.append(self._segment_queue.get_nowait())
            except queue.Empty:
                break
        return segments

    def set_audio_level(self, level: float):
        with self._level_lock:
            self._audio_level = level

    def get_audio_level(self) -> float:
        with self._level_lock:
            return self._audio_level

rec_state = RecordingState()
recording_thread: Optional[threading.Thread] = None
transcription_thread: Optional[threading.Thread] = None
recording_start_time: Optional[datetime] = None
recording_duration_final: Optional[float] = None
live_transcript: str = ""
live_total_segments: int = 0

# =============================================================================
# LIVE RECORDING WORKERS - ENHANCED FOR REAL-TIME
# =============================================================================
def recording_worker(rec_state: RecordingState, sample_rate: int, input_device=None):
    """Recording worker with device selection support"""
    if not LIVE_RECORDING_AVAILABLE:
        return
        
    def audio_callback(indata, frames, time_info, status):
        if status and "overflow" not in str(status).lower():
            print(f"[AUDIO] {status}")
        audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        rms = calculate_rms(audio_data)
        rec_state.set_audio_level(rms)
        rec_state.add_audio(audio_data.copy())
        rec_state.add_full_audio(audio_data.copy())

    try:
        # Get device info
        if input_device is not None:
            device_info = sd.query_devices(input_device, 'input')
        else:
            device_info = sd.query_devices(kind='input')
        
        device_name = device_info['name']
        
        print(f"[INFO] ═══════════════════════════════════════════════════")
        print(f"[INFO] Recording Configuration:")
        print(f"[INFO]   Device: {device_name}")
        print(f"[INFO]   Device ID: {input_device if input_device is not None else 'default'}")
        print(f"[INFO]   Sample Rate: {sample_rate} Hz")
        print(f"[INFO]   Channels: {AUDIO_CHANNELS}")
        print(f"[INFO] ═══════════════════════════════════════════════════")
        
        with sd.InputStream(
            channels=AUDIO_CHANNELS,
            samplerate=sample_rate,
            callback=audio_callback,
            dtype=np.float32,
            blocksize=AUDIO_BLOCKSIZE,
            latency="high",
            device=input_device,  # Use specified device
        ):
            print(f"[INFO] ✓ Live recording started from: {device_name}")
            while not rec_state.should_stop:
                time.sleep(0.05)
            print("[INFO] Live recording stopped")
    except Exception as e:
        print(f"[ERROR] Recording error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rec_state.mark_stopped()


def transcription_worker(rec_state: RecordingState, sample_rate: int, segment_interval: float):
    """Enhanced transcription worker with faster processing"""
    audio_buffer = []
    buffer_samples = 0
    last_transcription_time = time.time()
    segment_counter = 0

    print(f"[INFO] Transcription worker started (chunk interval: {segment_interval}s)")

    while True:
        chunks = rec_state.get_audio_chunks(timeout=0.05)  # Faster polling
        for chunk in chunks:
            audio_buffer.append(chunk)
            buffer_samples += len(chunk)

        current_time = time.time()
        time_elapsed = current_time - last_transcription_time

        # More aggressive transcription triggering
        should_transcribe = (
            (time_elapsed >= segment_interval and buffer_samples > 0) or
            (not rec_state.is_recording and buffer_samples > 0)
        )

        if should_transcribe and audio_buffer:
            audio_segment = np.concatenate(audio_buffer)
            audio_buffer = []
            buffer_samples = 0
            last_transcription_time = current_time

            min_samples = int(sample_rate * LIVE_MIN_AUDIO_LENGTH)
            if len(audio_segment) < min_samples:
                continue

            rms = calculate_rms(audio_segment)
            if rms < LIVE_SILENCE_THRESHOLD:
                # Skip silent segments
                continue

            segment_counter += 1
            
            # Process transcription in background to not block
            threading.Thread(
                target=process_live_transcription_segment,
                args=(audio_segment, segment_counter, rec_state),
                daemon=True
            ).start()

        if not rec_state.is_recording and len(audio_buffer) == 0:
            break

        time.sleep(0.02)  # Very short sleep for responsiveness

    print("[INFO] Transcription worker stopped")


def process_live_transcription_segment(audio_segment: np.ndarray, segment_id: int, rec_state: RecordingState):
    """
    Process audio segment with Hindi/English dual-pass detection
    FIXED: Increased timeout to 30s
    """
    try:
        # Resample if necessary
        if RECORDING_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
            num_samples = int(len(audio_segment) * WHISPER_SAMPLE_RATE / RECORDING_SAMPLE_RATE)
            audio_resampled = signal.resample(audio_segment, num_samples)
        else:
            audio_resampled = audio_segment
        
        # Calculate RMS
        rms_original = calculate_rms(audio_resampled)
        
        # Skip if too quiet
        if rms_original < LIVE_SILENCE_THRESHOLD:
            return
        
        # Normalize audio
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_normalized = audio_resampled / max_val
        else:
            return
        
        # Convert to int16
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Save with correct sample rate
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wavfile.write(tmp.name, WHISPER_SAMPLE_RATE, audio_int16)
            tmp_path = tmp.name
        
        print(f"[DEBUG] Seg {segment_id}: RMS={rms_original:.4f}")

        # Send to Whisper with dual-pass
        import requests
        with open(tmp_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            data = {
                'language': 'hi',  # Start with Hindi
                'beam_size': '5',
                'best_of': '5',
                'temperature': '0.0',
                'condition_on_previous_text': 'false',
                'word_timestamps': 'true',
                'task': 'transcribe',
                'compression_ratio_threshold': '2.4',
                'logprob_threshold': '-1.0',
                'no_speech_threshold': '0.6'
            }
            
            response = requests.post(
                f"{WHISPER_URL}/transcribe",
                files=files,
                data=data,
                timeout=TRANSCRIPTION_TIMEOUT  # FIXED: Use 30s timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                detected_language = result.get('language', 'hi')
                
                # Check if result looks like English
                if detected_language == 'hi' and text:
                    english_words = len([w for w in text.split() if w.isascii() and len(w) > 2])
                    total_words = len(text.split())
                    
                    # If >50% looks English, retry
                    if total_words > 0 and (english_words / total_words) > 0.5:
                        print(f"[DEBUG] Re-trying segment as English...")
                        
                        with open(tmp_path, 'rb') as f2:
                            files2 = {'file': ('audio.wav', f2, 'audio/wav')}
                            data2 = data.copy()
                            data2['language'] = 'en'
                            
                            response2 = requests.post(
                                f"{WHISPER_URL}/transcribe",
                                files=files2,
                                data=data2,
                                timeout=TRANSCRIPTION_TIMEOUT
                            )
                            
                            if response2.status_code == 200:
                                result2 = response2.json()
                                text2 = result2.get('text', '').strip()
                                
                                if len(text2) > len(text) * 0.8:
                                    text = text2
                                    detected_language = 'en'
                
                # Accept text
                if text and len(text) > 3:
                    hallucinations = ['thank you', 'thanks for watching', 'cool', 'gracias']
                    if text.lower().strip() not in hallucinations:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        rec_state.add_segment({
                            "timestamp": timestamp,
                            "text": text,
                            "language": detected_language,
                            "segment_id": segment_id
                        })
                        print(f"[LIVE] [{timestamp}] [{detected_language.upper()}] {text}")

        os.unlink(tmp_path)
        
    except requests.Timeout:
        print(f"[TIMEOUT] Seg {segment_id}: Transcription took >30s")
    except Exception as e:
        print(f"[ERROR] Seg {segment_id}: {e}")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="AI Pipeline Gateway (Enhanced)",
    version="2.0.0",
    description="Unified API for Whisper + NeMo + Llama with Live Recording"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# STARTUP EVENT
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Wait for all AI services to be ready"""
    print("[INFO] Starting AI Gateway...")
    print("[INFO] Checking AI service availability...")
    
    # Wait for all services
    services = [
        (WHISPER_URL, "Whisper Transcription"),
        (NEMO_URL, "NeMo Diarization"),
        (LLAMA_URL, "Llama Summarization"),
    ]
    
    for url, name in services:
        await wait_for_service(url, name)
    
    print("[INFO] ✓ All AI services are ready!")
    print("[INFO] 🚀 AI Gateway is ready to accept requests!")

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
async def root():
    return {
        "service": "AI Pipeline Gateway (Enhanced)",
        "version": "2.0.0",
        "status": "ready",
        "features": {
            "transcription": True,
            "diarization": True,
            "summarization": True,
            "live_recording": LIVE_RECORDING_AVAILABLE,
            "caching": True,
            "pdf_export": True,
            "live_chunk_duration": LIVE_CHUNK_DURATION
        }
    }

@app.get("/health")
async def health():
    services_health = {}
    
    async with httpx.AsyncClient() as client:
        try:
            whisper = await client.get(f"{WHISPER_URL}/health", timeout=5.0)
            services_health["whisper"] = {"status": "healthy", "details": whisper.json()}
        except Exception as e:
            services_health["whisper"] = {"status": "unhealthy", "error": str(e)}
        
        try:
            llama = await client.get(f"{LLAMA_URL}/health", timeout=5.0)
            services_health["llama"] = {"status": "healthy", "details": llama.json()}
        except Exception as e:
            services_health["llama"] = {"status": "unhealthy", "error": str(e)}
        
        try:
            nemo = await client.get(f"{NEMO_URL}/health", timeout=5.0)
            services_health["nemo"] = {"status": "healthy", "details": nemo.json()}
        except Exception as e:
            services_health["nemo"] = {"status": "unhealthy", "error": str(e)}
    
    all_healthy = all(s["status"] == "healthy" for s in services_health.values())
    
    return {
        "gateway": "healthy",
        "all_services_healthy": all_healthy,
        "live_recording_available": LIVE_RECORDING_AVAILABLE,
        "services": services_health
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{WHISPER_URL}/transcribe", files=files)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{NEMO_URL}/diarize", files=files)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/diarize")
async def debug_diarize(file: UploadFile = File(...)):
    """Debug endpoint to see raw diarization response"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{NEMO_URL}/diarize", files=files)
            
            print("=" * 80)
            print("RAW DIARIZATION RESPONSE:")
            print(response.text)
            print("=" * 80)
            
            return {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }
    except Exception as e:
        return {"error": str(e)}

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{LLAMA_URL}/summarize",
                json={"text": request.text, "max_length": request.max_length}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/full-pipeline")
async def full_pipeline(file: UploadFile = File(...), summarize_output: bool = True):
    file_content = await file.read()
    file_hash = file_cache.get_file_hash(file_content)
    
    cached = file_cache.get(file_hash)
    if cached:
        return {"filename": file.filename, "from_cache": True, **cached}
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Run transcription and diarization in parallel
            print("[INFO] Starting parallel transcription and diarization...")
            
            transcribe_task = client.post(
                f"{WHISPER_URL}/transcribe",
                files={"file": (file.filename, file_content, file.content_type)},
            )
            
            diarize_task = client.post(
                f"{NEMO_URL}/diarize",
                files={"file": (file.filename, file_content, file.content_type)},
            )
            
            # Wait for both to complete
            results = await asyncio.gather(
                transcribe_task,
                diarize_task,
                return_exceptions=True
            )
            
            transcribe_response, diarize_response = results
            
            # Check for errors
            if isinstance(transcribe_response, Exception):
                print(f"[ERROR] Transcription failed: {transcribe_response}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(transcribe_response)}")
            
            if isinstance(diarize_response, Exception):
                print(f"[ERROR] Diarization failed: {diarize_response}")
                raise HTTPException(status_code=500, detail=f"Diarization failed: {str(diarize_response)}")
            
            if transcribe_response.status_code != 200:
                raise HTTPException(
                    status_code=transcribe_response.status_code,
                    detail=f"Transcription failed: {transcribe_response.text}"
                )
            
            if diarize_response.status_code != 200:
                raise HTTPException(
                    status_code=diarize_response.status_code,
                    detail=f"Diarization failed: {diarize_response.text}"
                )
            
            transcription = transcribe_response.json()
            diarization = diarize_response.json()
            
            print(f"[INFO] ✓ Transcription completed: {len(transcription.get('segments', []))} segments")
            print(f"[INFO] ✓ Diarization completed: {len(diarization.get('speakers', []))} speaker segments")
            
            # Format diarized transcript
            formatted_transcript = format_diarized_transcript(transcription, diarization)
            
            # Count unique speakers
            speakers = diarization.get("speakers", [])
            unique_speakers = set(spk.get("speaker", "SPEAKER_00") for spk in speakers)
            num_speakers = len(unique_speakers)
            diarization["num_speakers"] = num_speakers
            
            print(f"[INFO] Starting summarization...")
            
            # Run summarization
            summary = None
            if summarize_output and transcription.get("text"):
                try:
                    summarize_response = await client.post(
                        f"{LLAMA_URL}/summarize",
                        json={"text": transcription["text"], "max_length": 300},
                    )
                    
                    if summarize_response.status_code == 200:
                        summary = summarize_response.json()
                        print(f"[INFO] ✓ Summarization completed")
                    else:
                        print(f"[WARN] Summarization failed with status {summarize_response.status_code}")
                        summary = {
                            "summary": "Summarization service unavailable",
                            "analysis": f"The summarization service returned an error: {summarize_response.text}"
                        }
                        
                except httpx.ConnectError as e:
                    print(f"[WARN] Summarization service not reachable: {e}")
                    summary = {
                        "summary": "Summarization service unavailable",
                        "analysis": "The summarization service is not currently available. Please try again later."
                    }
                except Exception as e:
                    print(f"[WARN] Summarization error: {e}")
                    summary = {
                        "summary": "Summarization failed",
                        "analysis": f"An error occurred during summarization: {str(e)}"
                    }
            
            result = {
                "transcription": transcription,
                "diarization": diarization,
                "formatted_transcript": formatted_transcript,
                "summary": summary
            }
            
            file_cache.set(file_hash, result, len(file_content))
            
            print(f"[INFO] ✓ Full pipeline completed for {file.filename}")
            
            return {"filename": file.filename, "from_cache": False, **result}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

# =============================================================================
# ENHANCED LIVE RECORDING ENDPOINTS
# =============================================================================
@app.post("/live/start")
async def live_start():
    global recording_thread, transcription_thread, recording_start_time
    global recording_duration_final, live_transcript, live_total_segments
    
    if not LIVE_RECORDING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Live recording not available")
    
    if rec_state.is_recording:
        raise HTTPException(status_code=400, detail="Already recording")

    rec_state.reset()
    rec_state.start()

    # Use faster chunk interval for live transcription
    recording_thread = threading.Thread(
        target=recording_worker,
        args=(rec_state, RECORDING_SAMPLE_RATE, INPUT_DEVICE_ID),
        daemon=True
    )
    transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(rec_state, RECORDING_SAMPLE_RATE, LIVE_CHUNK_DURATION),  # 2s chunks
        daemon=True
    )
    
    recording_thread.start()
    transcription_thread.start()

    recording_start_time = datetime.now()
    recording_duration_final = None
    live_transcript = ""
    live_total_segments = 0

    print("[INFO] Live recording started with 2-second chunk processing")
    
    return {
        "status": "recording_started",
        "chunk_duration": LIVE_CHUNK_DURATION,
        "sample_rate": RECORDING_SAMPLE_RATE,
        "device_id": INPUT_DEVICE_ID
    }


@app.post("/live/stop")
async def live_stop():
    """Stop live recording"""
    global recording_duration_final, recording_start_time

    if not rec_state.is_recording:
        raise HTTPException(status_code=400, detail="Not recording")

    rec_state.stop()

    if recording_start_time:
        recording_duration_final = (datetime.now() - recording_start_time).total_seconds()

    # Wait for transcription thread to finish processing
    if transcription_thread and transcription_thread.is_alive():
        print("[INFO] Waiting for remaining segments to be transcribed...")
        transcription_thread.join(timeout=10.0)

    print(f"[INFO] Recording stopped. Duration: {recording_duration_final:.2f}s")
    
    return {
        "status": "stopped",
        "duration": recording_duration_final,
        "total_segments": live_total_segments
    }


@app.get("/live/status", response_model=LiveStatusResponse)
async def live_status():
    """Enhanced live status with real-time updates"""
    global live_transcript, live_total_segments

    new_segments = rec_state.get_segments()
    latest_text = None
    
    # Add new segments to transcript
    for seg in new_segments:
        live_transcript += f"[{seg['timestamp']}] {seg['text']}\n"
        live_total_segments += 1
        latest_text = seg['text']

    elapsed = (
        (datetime.now() - recording_start_time).total_seconds()
        if recording_start_time and rec_state.is_recording
        else (recording_duration_final or 0.0)
    )

    return LiveStatusResponse(
        is_recording=rec_state.is_recording,
        duration=elapsed,
        word_count=count_words(live_transcript),
        total_segments=live_total_segments,
        latest_text=latest_text,
        transcript=live_transcript,
        audio_level=rec_state.get_audio_level()
    )


@app.post("/live/diarize")
async def live_diarize():
    """
    Perform diarization with Hindi/English only
    FIXED: Force Hindi language + better filtering + 30s timeout
    """
    global live_transcript
    
    if rec_state.is_recording:
        raise HTTPException(status_code=400, detail="Stop recording first")
    
    full_audio = rec_state.get_full_audio()
    if full_audio is None or len(full_audio) == 0:
        raise HTTPException(status_code=400, detail="No audio recorded")
    
    try:
        # Resample if necessary
        if RECORDING_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
            print(f"[INFO] Resampling from {RECORDING_SAMPLE_RATE}Hz to {WHISPER_SAMPLE_RATE}Hz...")
            num_samples = int(len(full_audio) * WHISPER_SAMPLE_RATE / RECORDING_SAMPLE_RATE)
            full_audio = signal.resample(full_audio, num_samples)
            print(f"[INFO] Resampled to {len(full_audio)} samples")
        
        # Normalize
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val
        
        # Save full audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_int16 = (full_audio * 32767).astype(np.int16)
            wavfile.write(tmp.name, WHISPER_SAMPLE_RATE, audio_int16)
            tmp_path = tmp.name
        
        duration = len(full_audio) / WHISPER_SAMPLE_RATE
        print(f"[INFO] Starting diarization on {duration:.2f}s of audio...")
        
        # Send to NeMo for diarization and Whisper for full transcription
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Diarization (NeMo only does speaker detection, not transcription)
            with open(tmp_path, 'rb') as f:
                files = {'file': ('recording.wav', f, 'audio/wav')}
                diarize_response = await client.post(f"{NEMO_URL}/diarize", files=files)
            
            if diarize_response.status_code != 200:
                raise HTTPException(
                    status_code=diarize_response.status_code,
                    detail=f"Diarization failed: {diarize_response.text}"
                )
            
            diarization = diarize_response.json()
            
            # Full transcription - FORCE Hindi to avoid Urdu detection
            with open(tmp_path, 'rb') as f:
                files = {'file': ('recording.wav', f, 'audio/wav')}
                data = {
                    'language': 'hi',  # FORCE Hindi (prevents Urdu detection)
                    'beam_size': '5',
                    'best_of': '5',
                    'temperature': '0.0',
                    'condition_on_previous_text': 'false',
                    'word_timestamps': 'true',
                    'task': 'transcribe',
                    'compression_ratio_threshold': '2.4',
                    'logprob_threshold': '-1.0',
                    'no_speech_threshold': '0.6'
                }
                transcribe_response = await client.post(
                    f"{WHISPER_URL}/transcribe",
                    files=files,
                    data=data
                )
            
            if transcribe_response.status_code == 200:
                transcription = transcribe_response.json()
                
                # POST-PROCESS: Check for English segments
                segments = transcription.get("segments", [])
                
                # If mostly English words detected, mark segment as English
                for seg in segments:
                    text = seg.get("text", "")
                    if text:
                        words = text.split()
                        english_ratio = len([w for w in words if w.isascii()]) / len(words) if words else 0
                        
                        # If segment is >70% English, mark it
                        if english_ratio > 0.7:
                            seg['language'] = 'en'
                        else:
                            seg['language'] = 'hi'
                
                print(f"[INFO] Transcribed {len(segments)} segments (Hindi with English code-switching)")
            else:
                transcription = {"segments": [], "text": live_transcript}
        
        os.unlink(tmp_path)
        
        # Format diarized transcript
        formatted_transcript = format_diarized_transcript(transcription, diarization)
        
        # Update live transcript
        live_transcript = formatted_transcript
        
        speakers = diarization.get("speakers", [])
        unique_speakers = set(spk.get("speaker", "SPEAKER_00") for spk in speakers)
        num_speakers = len(unique_speakers)
        
        print(f"[INFO] Diarization complete: {num_speakers} speakers detected")
        
        return {
            "status": "completed",
            "formatted_transcript": formatted_transcript,
            "num_speakers": num_speakers,
            "diarization": diarization
        }
        
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")


@app.post("/live/reset")
async def live_reset():
    """Reset live recording state"""
    global live_transcript, live_total_segments, recording_start_time, recording_duration_final
    
    if rec_state.is_recording:
        raise HTTPException(status_code=400, detail="Stop recording first")
    
    rec_state.reset()
    live_transcript = ""
    live_total_segments = 0
    recording_start_time = None
    recording_duration_final = None
    
    print("[INFO] Live recording state reset")
    
    return {"status": "reset"}


@app.post("/cache/clear")
async def clear_cache():
    file_cache.clear()
    return {"status": "cleared"}

@app.get("/cache/stats")
async def cache_stats():
    return {
        "size_mb": file_cache.current_size / 1024 / 1024,
        "files": len(file_cache.cache)
    }

@app.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    if not req.content:
        raise HTTPException(status_code=400, detail="No content")
    buffer = build_pdf_buffer(req.content, req.title or "Transcript")
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="transcript.pdf"'})

@app.post("/export/complete-txt")
async def export_complete_txt(req: CompleteExportRequest):
    """Generate a complete TXT file with all sections"""
    content_parts = []
    
    # Header
    content_parts.append("=" * 80)
    content_parts.append(f"TRANSCRIPTION REPORT: {req.filename}")
    content_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_parts.append(f"Speakers Detected: {req.speaker_count}")
    content_parts.append("=" * 80)
    content_parts.append("")
    
    # Section 1: Diarized Transcript
    content_parts.append("DIARIZED TRANSCRIPT")
    content_parts.append("-" * 80)
    content_parts.append(req.formatted_transcript)
    content_parts.append("")
    
    # Section 2: AI Summary
    if req.summary:
        content_parts.append("=" * 80)
        content_parts.append("AI SUMMARY")
        content_parts.append("-" * 80)
        content_parts.append(req.summary)
        content_parts.append("")
    
    # Section 3: Raw Transcript
    content_parts.append("=" * 80)
    content_parts.append("RAW TRANSCRIPT")
    content_parts.append("-" * 80)
    content_parts.append(req.raw_transcript)
    content_parts.append("")
    
    # Footer
    content_parts.append("=" * 80)
    content_parts.append("© 2026 AngelBot.AI · Whisper · NeMo · Llama")
    content_parts.append("=" * 80)
    
    complete_content = "\n".join(content_parts)
    
    buffer = BytesIO(complete_content.encode('utf-8'))
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{req.filename}_complete.txt"'
        }
    )

@app.post("/export/complete-pdf")
async def export_complete_pdf(req: CompleteExportRequest):
    """Generate a complete PDF file with all sections"""
    buffer = BytesIO()
    
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        x, y = 50, height - 50
        
        def check_page_break():
            nonlocal y
            if y < 80:
                c.showPage()
                y = height - 50
                return True
            return False
        
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, y, "TRANSCRIPTION REPORT")
        y -= 20
        
        c.setFont("Helvetica", 10)
        c.drawString(x, y, f"File: {req.filename}")
        y -= 15
        c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 15
        c.drawString(x, y, f"Speakers Detected: {req.speaker_count}")
        y -= 30
        
        # Section 1: Diarized Transcript
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, "DIARIZED TRANSCRIPT")
        y -= 20
        
        c.setFont("Courier", 8)
        for line in req.formatted_transcript.split('\n'):
            check_page_break()
            # Wrap long lines
            if len(line) > 90:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) < 90:
                        current_line += word + " "
                    else:
                        c.drawString(x, y, current_line.strip())
                        y -= 12
                        check_page_break()
                        current_line = word + " "
                if current_line:
                    c.drawString(x, y, current_line.strip())
                    y -= 12
            else:
                c.drawString(x, y, line[:90])
                y -= 12
        
        y -= 20
        
        # Section 2: AI Summary
        if req.summary:
            check_page_break()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x, y, "AI SUMMARY")
            y -= 20
            
            c.setFont("Helvetica", 10)
            for line in req.summary.split('\n'):
                check_page_break()
                c.drawString(x, y, line[:80])
                y -= 14
            
            y -= 20
        
        # Section 3: Raw Transcript
        check_page_break()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, "RAW TRANSCRIPT")
        y -= 20
        
        c.setFont("Helvetica", 9)
        for line in req.raw_transcript.split('\n'):
            check_page_break()
            c.drawString(x, y, line[:85])
            y -= 12
        
        c.showPage()
        c.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{req.filename}_complete.pdf"'
            }
        )
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)