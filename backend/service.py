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
import re
from scipy import signal
import requests

import numpy as np
from scipy.io import wavfile

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# Try to import transliteration library
try:
    from indictrans import Transliterator
    trn = Transliterator(source='hin', target='eng', build_lookup=True)
    TRANSLITERATION_AVAILABLE = True
    print("[INFO] ✓ Transliteration library loaded")
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    trn = None
    print("[WARN] indic-transliteration not available, transliteration disabled")

# Try to import translation library
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATION_AVAILABLE = True
    print("[INFO] ✓ Google Translator loaded")
except ImportError:
    TRANSLATION_AVAILABLE = False
    translator = None
    print("[WARN] googletrans not available, translation disabled")

# =============================================================================
# CONFIG
# =============================================================================
WHISPER_URL = "http://speech-unified:8000"
LLAMA_URL = "http://llama-summarizer:8001"
NEMO_URL = "http://nemo-diarizer:8003"

TIMEOUT = 600.0
TRANSCRIPTION_TIMEOUT = 30.0

# Audio settings
WHISPER_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCKSIZE = 8192

# Legacy settings (for file upload)
SEGMENT_INTERVAL = 5.0
MIN_AUDIO_LENGTH = 0.5
SILENCE_THRESHOLD = 0.01

# Live recording settings
LIVE_CHUNK_DURATION = 2.0
LIVE_MIN_AUDIO_LENGTH = 0.8
LIVE_SILENCE_THRESHOLD = 0.005
LIVE_VAD_ENABLED = True

FILE_CACHE_SIZE_MB = 500

# Translation settings (IMPROVED)
TRANSLATION_CHUNK_SIZE = 2000
TRANSLATION_TIMEOUT = 300.0
TRANSLATION_MAX_RETRIES = 3

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
INPUT_DEVICE_ID = 4

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
    raw_transcript: Optional[str] = ""
    filename: str
    speaker_count: int

class TranslateTextRequest(BaseModel):
    text: str
    source_lang: str = "hi"
    target_lang: str = "en"

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
# HALLUCINATION FILTERING
# =============================================================================
def filter_hallucinations(transcript: str) -> str:
    """Remove common Whisper hallucinations before processing"""
    hallucination_patterns = [
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*thanks for watching',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*please give it a like',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*subscribe to my channel',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*if you enjoyed this video',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*consider subscribing',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*thank you\s*$',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*cool\s*$',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*gracias\s*$',
        r'\[[\d:]+\]\s*SPEAKER_\d+:\s*bye bye\s*$',
    ]
    
    lines = transcript.split('\n')
    filtered_lines = []
    
    for line in lines:
        is_hallucination = False
        line_lower = line.lower()
        
        for pattern in hallucination_patterns:
            if re.search(pattern, line_lower):
                is_hallucination = True
                print(f"[FILTERED] Hallucination: {line[:60]}...")
                break
        
        if not is_hallucination:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def clean_llama_translation_output(text: str) -> str:
    """Remove meta-text that Llama sometimes adds"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        lower = line.lower().strip()
        
        # Skip meta-commentary
        skip_patterns = [
            'here is the translation',
            'note:',
            'translation:',
            'i have translated',
            "i've translated",
            'the translation is',
            'translated text:',
            'output:',
        ]
        
        should_skip = any(lower.startswith(pattern) for pattern in skip_patterns)
        
        if should_skip:
            print(f"[CLEANUP] Removed meta-text: {line[:60]}...")
            continue
        
        # Skip empty lines
        if not line.strip():
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# =============================================================================
# TRANSLATION FUNCTIONS (FULLY IMPROVED WITH RETRY LOGIC)
# =============================================================================

async def translate_chunk_with_retry(chunk: str, source_lang: str, target_lang: str, chunk_num: int, total_chunks: int, max_retries: int = 3) -> str:
    """
    Translate a single chunk with retry logic and validation
    Returns translated text or original text if all retries fail
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=TRANSLATION_TIMEOUT) as client:
                print(f"[TRANSLATE] Chunk {chunk_num}/{total_chunks} - Attempt {attempt + 1}/{max_retries}")
                print(f"[TRANSLATE] Chunk size: {len(chunk)} chars")
                
                response = await client.post(
                    f"{LLAMA_URL}/translate",
                    json={
                        "text": chunk,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "max_length": len(chunk) * 3,
                        "temperature": 0.2
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated = result.get("translated_text", "")
                    
                    # Clean up Llama meta-text
                    translated = clean_llama_translation_output(translated)
                    
                    # Validate output
                    if translated and len(translated.strip()) > 10:
                        print(f"[TRANSLATE] ✓ Chunk {chunk_num}/{total_chunks} completed ({len(translated)} chars)")
                        return translated
                    else:
                        print(f"[WARN] Output too short or empty for chunk {chunk_num}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                else:
                    print(f"[WARN] HTTP {response.status_code} for chunk {chunk_num}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
        
        except httpx.TimeoutException:
            print(f"[ERROR] Timeout on chunk {chunk_num}, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
                continue
        except Exception as e:
            print(f"[ERROR] Chunk {chunk_num}, attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
    
    # All retries failed
    print(f"[ERROR] ❌ All {max_retries} attempts failed for chunk {chunk_num}")
    print(f"[ERROR] Using original text as fallback")
    return chunk


async def translate_text_with_llama(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using Llama model via /translate endpoint
    FULLY IMPROVED: Larger chunks, retry logic, detailed logging
    """
    if not text or len(text.strip()) == 0:
        return text
    
    try:
        if source_lang == target_lang:
            return text
        
        # Split into LARGER chunks (now 2000 chars instead of 500)
        chunks = []
        if len(text) > TRANSLATION_CHUNK_SIZE:
            lines = text.split('\n')
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk) + len(line) + 1 < TRANSLATION_CHUNK_SIZE:
                    current_chunk += line + "\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]
        
        print(f"\n[TRANSLATE] ═══════════════════════════════════════════")
        print(f"[TRANSLATE] Total text: {len(text)} chars")
        print(f"[TRANSLATE] Chunk size: {TRANSLATION_CHUNK_SIZE} chars")
        print(f"[TRANSLATE] Number of chunks: {len(chunks)}")
        print(f"[TRANSLATE] Direction: {source_lang} → {target_lang}")
        print(f"[TRANSLATE] Max retries per chunk: {TRANSLATION_MAX_RETRIES}")
        print(f"[TRANSLATE] ═══════════════════════════════════════════\n")
        
        # Translate chunks with retry logic
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            
            print(f"\n[TRANSLATE] --- Processing Chunk {chunk_num}/{len(chunks)} ---")
            print(f"[TRANSLATE] Preview: {chunk[:100]}...")
            
            translated = await translate_chunk_with_retry(
                chunk, 
                source_lang, 
                target_lang,
                chunk_num,
                len(chunks),
                max_retries=TRANSLATION_MAX_RETRIES
            )
            
            translated_chunks.append(translated)
        
        final_translation = "\n".join(translated_chunks)
        
        print(f"\n[TRANSLATE] ═══════════════════════════════════════════")
        print(f"[TRANSLATE] ✓ Translation completed successfully")
        print(f"[TRANSLATE] Input length: {len(text)} chars")
        print(f"[TRANSLATE] Output length: {len(final_translation)} chars")
        print(f"[TRANSLATE] Ratio: {len(final_translation)/len(text):.2f}x")
        print(f"[TRANSLATE] ═══════════════════════════════════════════\n")
        
        return final_translation
    
    except Exception as e:
        print(f"[ERROR] Translation pipeline failed: {e}")
        traceback.print_exc()
        return text


async def translate_transcript_preserve_format(transcript: str, source_lang: str, target_lang: str) -> str:
    """
    Translate transcript while preserving timestamps and speaker labels
    IMPROVED: Translate line-by-line for perfect alignment
    """
    try:
        # FILTER HALLUCINATIONS FIRST (with timestamps)
        transcript = filter_hallucinations(transcript)
        
        lines = transcript.strip().split('\n')
        translated_lines = []
        
        print(f"[TRANSLATE] Processing {len(lines)} lines individually...")
        
        # Translate each line separately to maintain exact 1:1 mapping
        for i, line in enumerate(lines):
            if not line.strip():
                translated_lines.append("")
                continue
            
            # Parse: [HH:MM:SS] SPEAKER_X: text
            if ']' in line and ':' in line:
                timestamp_match = line.split(']', 1)
                if len(timestamp_match) == 2:
                    timestamp = timestamp_match[0] + ']'
                    rest = timestamp_match[1].strip()
                    
                    if ':' in rest and 'SPEAKER_' in rest:
                        speaker_match = rest.split(':', 1)
                        if len(speaker_match) == 2:
                            speaker = speaker_match[0].strip()
                            text = speaker_match[1].strip()
                            
                            # Skip if text is too short
                            if len(text.strip()) < 3:
                                translated_lines.append(line)
                                continue
                            
                            # Translate this single line
                            print(f"[TRANSLATE] Line {i+1}/{len(lines)}: {text[:50]}...")
                            
                            try:
                                async with httpx.AsyncClient(timeout=60.0) as client:
                                    response = await client.post(
                                        f"{LLAMA_URL}/translate",
                                        json={
                                            "text": text,
                                            "source_lang": source_lang,
                                            "target_lang": target_lang,
                                            "max_length": len(text) * 3,
                                            "temperature": 0.2
                                        }
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        translated_text = result.get("translated_text", "").strip()
                                        
                                        # Clean up meta-text
                                        translated_text = clean_llama_translation_output(translated_text)
                                        
                                        # Remove any accidental timestamp/speaker from Llama
                                        if ']' in translated_text:
                                            translated_text = translated_text.split(']', 1)[-1].strip()
                                        if 'SPEAKER_' in translated_text and ':' in translated_text:
                                            parts = translated_text.split(':', 1)
                                            if len(parts) > 1:
                                                translated_text = parts[1].strip()
                                        
                                        # Take only the first line if Llama returned multiple
                                        if '\n' in translated_text:
                                            translated_text = translated_text.split('\n')[0].strip()
                                        
                                        # Reconstruct with original structure
                                        reconstructed = f"{timestamp} {speaker}: {translated_text}"
                                        translated_lines.append(reconstructed)
                                        
                                        print(f"[TRANSLATE] ✓ Line {i+1}/{len(lines)}: {translated_text[:50]}...")
                                    else:
                                        print(f"[WARN] Line {i+1} failed, using original")
                                        translated_lines.append(line)
                                
                            except Exception as e:
                                print(f"[ERROR] Line {i+1} failed: {e}")
                                translated_lines.append(line)
                            
                            continue
            
            # Couldn't parse, keep original
            translated_lines.append(line)
        
        result = '\n'.join(translated_lines)
        print(f"[TRANSLATE] ✓ Line-by-line translation completed")
        return result
    
    except Exception as e:
        print(f"[ERROR] Transcript translation failed: {e}")
        traceback.print_exc()
        return transcript

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


def is_hallucination(text: str) -> bool:
    """Enhanced hallucination detection for English and Hindi"""
    if not text or len(text.strip()) < 3:
        return False
    
    # English hallucination phrases
    english_hallucinations = [
        'thank you', 'thanks for watching', 'subscribe', 'like and subscribe',
        'cool', 'gracias', 'bye bye', 'see you next time', 'please subscribe',
        'if you enjoyed', 'hit the bell', 'notification'
    ]
    
    # Hindi hallucination phrases and single-word artifacts
    hindi_hallucinations = [
        'कि',      # Single "ki"
        'है',      # Single "hai"  
        'तो',      # Single "to"
        'और',      # Single "aur"
        'यह',      # Single "yeh"
        'वह',      # Single "vah"
        'में',     # Single "mein"
        'के',      # Single "ke"
        'का',      # Single "ka"
        'की',      # Single "ki"
        'से',      # Single "se"
        'पर',      # Single "par"
    ]
    
    text_stripped = text.strip()
    text_lower = text.lower().strip()
    
    # Check English hallucinations
    for phrase in english_hallucinations:
        if phrase in text_lower:
            return True
    
    # Check Hindi single-word artifacts
    if text_stripped in hindi_hallucinations:
        print(f"[HALLUCINATION] Single-word Hindi artifact: '{text_stripped}'")
        return True
    
    # Check if text is just punctuation or single char
    if len(text_stripped) <= 2:
        return True
    
    # Check for excessive repetition
    words = text.split()
    if len(words) >= 3:
        from collections import Counter
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if count >= 3 and count / len(words) > 0.4:
                print(f"[HALLUCINATION] Word '{word}' repeated {count}/{len(words)} times")
                return True
    
    return False




def transliterate_hindi_to_english(text: str) -> str:
    """Transliterate Hindi Devanagari to Roman script"""
    if not TRANSLITERATION_AVAILABLE or not trn:
        return text
    
    try:
        if not text or len(text.strip()) == 0:
            return text
        
        transliterated = trn.transform(text)
        return transliterated
    
    except Exception as e:
        print(f"[WARN] Transliteration failed: {e}")
        return text


def transliterate_transcript_with_speakers(formatted_transcript: str) -> str:
    """Transliterate diarized transcript while preserving structure"""
    if not TRANSLITERATION_AVAILABLE or not trn:
        return formatted_transcript
    
    try:
        lines = formatted_transcript.split('\n')
        transliterated_lines = []
        
        for line in lines:
            if not line.strip():
                transliterated_lines.append(line)
                continue
            
            if ']' in line and ':' in line:
                timestamp_part = line.split(']', 1)[0] + ']'
                rest = line.split(']', 1)[1].strip()
                
                if ':' in rest:
                    speaker_label = rest.split(':', 1)[0]
                    text = rest.split(':', 1)[1].strip()
                    
                    transliterated_text = transliterate_hindi_to_english(text)
                    new_line = f"{timestamp_part} {speaker_label}: {transliterated_text}"
                    transliterated_lines.append(new_line)
                else:
                    transliterated_text = transliterate_hindi_to_english(rest)
                    new_line = f"{timestamp_part} {transliterated_text}"
                    transliterated_lines.append(new_line)
            else:
                transliterated_lines.append(transliterate_hindi_to_english(line))
        
        return '\n'.join(transliterated_lines)
    
    except Exception as e:
        print(f"[ERROR] Transcript transliteration failed: {e}")
        return formatted_transcript


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
        
        if is_hallucination(text):
            print(f"[FILTERED] Hallucination detected: {text[:50]}...")
            continue
        
        words = trans_seg.get("words", [])
        
        if not words or len(words) == 0:
            speaker = find_speaker_for_time(start_time, end_time, speaker_segments)
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            formatted_lines.append(f"[{time_str}] {speaker}: {text}")
        else:
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
            c.drawString(x, y, line[:100])
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
# LIVE RECORDING STATE
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
        self._all_segments: List[dict] = []

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
        self._all_segments = []

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
        self._all_segments.append(segment)

    def get_all_segments(self) -> List[dict]:  # ADD THIS METHOD
        """Get all segments accumulated during recording"""
        return self._all_segments.copy()

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
# LIVE RECORDING WORKERS
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
            device=input_device,
        ):
            print(f"[INFO] ✓ Live recording started from: {device_name}")
            while not rec_state.should_stop:
                time.sleep(0.05)
            print("[INFO] Live recording stopped")
    except Exception as e:
        print(f"[ERROR] Recording error: {e}")
        traceback.print_exc()
    finally:
        rec_state.mark_stopped()


def transcription_worker(rec_state: RecordingState, sample_rate: int, segment_interval: float):
    """Enhanced transcription worker"""
    audio_buffer = []
    buffer_samples = 0
    last_transcription_time = time.time()
    segment_counter = 0

    print(f"[INFO] Transcription worker started (chunk interval: {segment_interval}s)")

    while True:
        chunks = rec_state.get_audio_chunks(timeout=0.05)
        for chunk in chunks:
            audio_buffer.append(chunk)
            buffer_samples += len(chunk)

        current_time = time.time()
        time_elapsed = current_time - last_transcription_time

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
                continue

            segment_counter += 1
            
            threading.Thread(
                target=process_live_transcription_segment,
                args=(audio_segment, segment_counter, rec_state),
                daemon=True
            ).start()

        if not rec_state.is_recording and len(audio_buffer) == 0:
            break

        time.sleep(0.02)

    print("[INFO] Transcription worker stopped")


def fallback_single_speaker_diarization(existing_segments: list, generate_summary: bool):
    """
    Fallback when NeMo diarization fails - assign all to SPEAKER_0
    """
    logger.info("[FALLBACK] Using single-speaker mode")
    
    formatted_transcript = ""
    
    for seg in existing_segments:
        if is_hallucination(seg['text']):
            continue
        
        formatted_transcript += f"[{seg['timestamp']}] SPEAKER_0: {seg['text']}\n"
    
    # Generate summary
    summary_result = None
    
    if generate_summary:
        try:
            logger.info("[FALLBACK] Generating MoM...")
            
            summary_response = requests.post(
                f"{LLAMA_SUMMARIZER_URL}/summarize",
                json={"text": formatted_transcript, "max_length": 4000, "temperature": 0.01},
                timeout=300
            )
            
            if summary_response.status_code == 200:
                summary_result = summary_response.json()
        except Exception as e:
            logger.error(f"[FALLBACK] Summary failed: {e}")
    
    return {
        "formatted_transcript": formatted_transcript,
        "num_speakers": 1,
        "summary": summary_result.get('analysis') if summary_result else None,
        "segments_count": len(existing_segments),
        "fallback_mode": True
    }


def normalize_speaker_labels(transcript: str) -> str:
    """Fix SPEAKER_00 → SPEAKER_0 and ensure consistent numbering"""
    import re
    
    lines = transcript.split('\n')
    normalized_lines = []
    
    speaker_map = {}
    next_speaker_num = 0
    
    for line in lines:
        if not line.strip():
            continue
        
        if 'SPEAKER_' in line:
            match = re.search(r'SPEAKER_(\d+)', line)
            if match:
                original_speaker = f"SPEAKER_{match.group(1)}"
                
                if original_speaker not in speaker_map:
                    speaker_map[original_speaker] = f"SPEAKER_{next_speaker_num}"
                    next_speaker_num += 1
                
                normalized_speaker = speaker_map[original_speaker]
                line = line.replace(original_speaker, normalized_speaker)
        
        normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)




def process_live_transcription_segment(audio_segment: np.ndarray, segment_id: int, rec_state: RecordingState):
    """
    Enhanced transcription with accurate language detection
    - English speech → English text
    - Hindi speech → Hindi (Devanagari) text  
    - Hinglish speech → Mixed text (as spoken)
    """
    try:
        # Resample to 16kHz for Whisper
        if RECORDING_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
            num_samples = int(len(audio_segment) * WHISPER_SAMPLE_RATE / RECORDING_SAMPLE_RATE)
            audio_resampled = signal.resample(audio_segment, num_samples)
        else:
            audio_resampled = audio_segment
        
        rms_original = calculate_rms(audio_resampled)
        
        # Filter silence
        if rms_original < LIVE_SILENCE_THRESHOLD:
            return
        
        # Normalize audio
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_normalized = audio_resampled / max_val
        else:
            return
        
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Save as temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wavfile.write(tmp.name, WHISPER_SAMPLE_RATE, audio_int16)
            tmp_path = tmp.name
        
        print(f"[DEBUG] Seg {segment_id}: RMS={rms_original:.4f}")

        import requests
        
        # =====================================================================
        # STEP 1: AUTO-DETECT LANGUAGE (No language specified)
        # =====================================================================
        with open(tmp_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            
            # First pass: Auto-detect language
            detect_data = {
                'beam_size': '5',
                'best_of': '5',
                'temperature': '0.0',
                'condition_on_previous_text': 'true',
                'word_timestamps': 'true',
                'task': 'transcribe',
                'compression_ratio_threshold': '2.0',
                'logprob_threshold': '-0.8',
                'no_speech_threshold': '0.7',
                # NO language parameter - let Whisper detect
            }
            
            detect_response = requests.post(
                f"{WHISPER_URL}/transcribe",
                files=files,
                data=detect_data,
                timeout=TRANSCRIPTION_TIMEOUT
            )
            
            if detect_response.status_code != 200:
                print(f"[ERROR] Seg {segment_id}: Detection failed")
                os.unlink(tmp_path)
                return
            
            detect_result = detect_response.json()
            detected_language = detect_result.get('language', 'en')
            detected_text = detect_result.get('text', '').strip()
            
            print(f"[DETECT] Seg {segment_id}: Detected={detected_language}, Text={detected_text[:50]}...")
        
        # =====================================================================
        # STEP 2: ANALYZE TEXT TO DETERMINE IF IT'S MIXED LANGUAGE
        # =====================================================================
        def analyze_language_mix(text: str) -> dict:
            """
            Analyze text to determine language composition
            Returns: {
                'is_mixed': bool,
                'english_ratio': float,
                'hindi_ratio': float,
                'dominant': 'en' or 'hi'
            }
            """
            if not text:
                return {'is_mixed': False, 'english_ratio': 0, 'hindi_ratio': 0, 'dominant': 'en'}
            
            words = text.split()
            total_words = len(words)
            
            if total_words == 0:
                return {'is_mixed': False, 'english_ratio': 0, 'hindi_ratio': 0, 'dominant': 'en'}
            
            # Count English words (ASCII characters)
            english_words = sum(1 for word in words if word and all(ord(c) < 128 for c in word if c.isalpha()))
            
            # Count Hindi words (Devanagari script: U+0900 to U+097F)
            hindi_words = sum(1 for word in words if word and any('\u0900' <= c <= '\u097F' for c in word))
            
            english_ratio = english_words / total_words
            hindi_ratio = hindi_words / total_words
            
            # Mixed if both languages present significantly
            is_mixed = english_ratio > 0.15 and hindi_ratio > 0.15
            
            dominant = 'hi' if hindi_ratio > english_ratio else 'en'
            
            return {
                'is_mixed': is_mixed,
                'english_ratio': english_ratio,
                'hindi_ratio': hindi_ratio,
                'dominant': dominant
            }
        
        language_analysis = analyze_language_mix(detected_text)
        
        print(f"[ANALYSIS] Seg {segment_id}: Mixed={language_analysis['is_mixed']}, "
              f"EN={language_analysis['english_ratio']:.2f}, "
              f"HI={language_analysis['hindi_ratio']:.2f}, "
              f"Dominant={language_analysis['dominant']}")
        
        # =====================================================================
        # STEP 3: DECIDE TRANSCRIPTION STRATEGY
        # =====================================================================
        
        final_text = detected_text
        final_language = detected_language
        
        # Case 1: Pure English (or English-dominant with very little Hindi)
        if language_analysis['dominant'] == 'en' and language_analysis['hindi_ratio'] < 0.1:
            print(f"[STRATEGY] Seg {segment_id}: Pure English detected")
            
            # Re-transcribe with English model for better accuracy
            with open(tmp_path, 'rb') as f:
                files = {'file': ('audio.wav', f, 'audio/wav')}
                english_data = detect_data.copy()
                english_data['language'] = 'en'
                
                english_response = requests.post(
                    f"{WHISPER_URL}/transcribe",
                    files=files,
                    data=english_data,
                    timeout=TRANSCRIPTION_TIMEOUT
                )
                
                if english_response.status_code == 200:
                    english_result = english_response.json()
                    final_text = english_result.get('text', '').strip()
                    final_language = 'en'
                    print(f"[ENGLISH] Seg {segment_id}: {final_text[:50]}...")
        
        # Case 2: Pure Hindi (or Hindi-dominant with very little English)
        elif language_analysis['dominant'] == 'hi' and language_analysis['english_ratio'] < 0.1:
            print(f"[STRATEGY] Seg {segment_id}: Pure Hindi detected")
            
            # Re-transcribe with Hindi model for better accuracy
            with open(tmp_path, 'rb') as f:
                files = {'file': ('audio.wav', f, 'audio/wav')}
                hindi_data = detect_data.copy()
                hindi_data['language'] = 'hi'
                
                hindi_response = requests.post(
                    f"{WHISPER_URL}/transcribe",
                    files=files,
                    data=hindi_data,
                    timeout=TRANSCRIPTION_TIMEOUT
                )
                
                if hindi_response.status_code == 200:
                    hindi_result = hindi_response.json()
                    final_text = hindi_result.get('text', '').strip()
                    final_language = 'hi'
                    print(f"[HINDI] Seg {segment_id}: {final_text[:50]}...")
        
        # Case 3: Mixed/Hinglish (significant presence of both languages)
        else:
            print(f"[STRATEGY] Seg {segment_id}: Hinglish/Mixed detected")
            
            # Use the auto-detected result (it handles Hinglish best)
            # But try both models and choose the one with better quality
            
            with open(tmp_path, 'rb') as f:
                files = {'file': ('audio.wav', f, 'audio/wav')}
                
                # Try Hindi model (better for Hinglish)
                hindi_data = detect_data.copy()
                hindi_data['language'] = 'hi'
                
                hindi_response = requests.post(
                    f"{WHISPER_URL}/transcribe",
                    files=files,
                    data=hindi_data,
                    timeout=TRANSCRIPTION_TIMEOUT
                )
                
                if hindi_response.status_code == 200:
                    hindi_result = hindi_response.json()
                    hindi_text = hindi_result.get('text', '').strip()
                    
                    # Compare: Use the result with more mixed content
                    hindi_analysis = analyze_language_mix(hindi_text)
                    
                    # If Hindi model preserves both languages better, use it
                    if hindi_analysis['is_mixed'] or len(hindi_text) > len(detected_text) * 0.9:
                        final_text = hindi_text
                        final_language = 'hinglish'
                        print(f"[HINGLISH] Seg {segment_id}: {final_text[:50]}...")
                    else:
                        # Use original auto-detected result
                        final_text = detected_text
                        final_language = 'hinglish'
                        print(f"[HINGLISH-AUTO] Seg {segment_id}: {final_text[:50]}...")
        
        # =====================================================================
        # STEP 4: FILTER AND VALIDATE
        # =====================================================================
        
        if not final_text or len(final_text) < 3:
            print(f"[FILTERED] Seg {segment_id}: Text too short")
            os.unlink(tmp_path)
            return
        
        # Filter hallucinations
        if is_hallucination(final_text):
            print(f"[FILTERED] Seg {segment_id}: Hallucination detected")
            os.unlink(tmp_path)
            return
        
        # Filter pure noise
        if not any(c.isalnum() for c in final_text):
            print(f"[FILTERED] Seg {segment_id}: Pure noise")
            os.unlink(tmp_path)
            return
        
        # =====================================================================
        # STEP 5: ADD TO TRANSCRIPT
        # =====================================================================
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        rec_state.add_segment({
            "timestamp": timestamp,
            "text": final_text,
            "language": final_language,
            "segment_id": segment_id
        })
        
        # Display with language tag
        lang_display = {
            'en': '🇬🇧 EN',
            'hi': '🇮🇳 HI',
            'hinglish': '🌐 MIX'
        }.get(final_language, final_language.upper())
        
        print(f"[LIVE] [{timestamp}] [{lang_display}] {final_text}")
        
        os.unlink(tmp_path)
        
    except requests.Timeout:
        print(f"[TIMEOUT] Seg {segment_id}: Transcription took >30s")
    except Exception as e:
        print(f"[ERROR] Seg {segment_id}: {e}")
        import traceback
        traceback.print_exc()


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
    
    services = [
        (WHISPER_URL, "Whisper Transcription"),
        (NEMO_URL, "NeMo Diarization"),
        (LLAMA_URL, "Llama Summarization + Translation"),
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
            "translation": True,
            "live_recording": LIVE_RECORDING_AVAILABLE,
            "caching": True,
            "pdf_export": True,
            "translation_chunk_size": TRANSLATION_CHUNK_SIZE,
            "translation_max_retries": TRANSLATION_MAX_RETRIES
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


# =============================================================================
# TRANSLATION ENDPOINT (IMPROVED)
# =============================================================================

@app.post("/translate-text")
async def translate_text_endpoint(request: TranslateTextRequest):
    """
    Translate text using Llama model with retry logic
    Preserves speaker labels and timestamps
    Filters hallucinations
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Filter hallucinations BEFORE translation
        cleaned_text = filter_hallucinations(request.text)
        
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No valid content after filtering")
        
        # Check if diarized format
        has_speaker_labels = "[" in cleaned_text and "SPEAKER_" in cleaned_text
        
        print(f"[TRANSLATE] Starting translation ({request.source_lang} → {request.target_lang})")
        print(f"[TRANSLATE] Text length: {len(cleaned_text)} chars")
        print(f"[TRANSLATE] Has speaker labels: {has_speaker_labels}")
        
        if has_speaker_labels:
            translated = await translate_transcript_preserve_format(
                cleaned_text,
                request.source_lang,
                request.target_lang
            )
        else:
            translated = await translate_text_with_llama(
                cleaned_text,
                request.source_lang,
                request.target_lang
            )
        
        print(f"[TRANSLATE] ✓ Translation completed ({len(translated)} chars)")
        
        return {
            "translated_text": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/full-pipeline")
async def full_pipeline(file: UploadFile = File(...), summarize_output: bool = True, transliterate: bool = True):
    """Full pipeline: Transcription + Diarization + Summary + Transliteration"""
    file_content = await file.read()
    file_hash = file_cache.get_file_hash(file_content)
    
    cached = file_cache.get(file_hash)
    if cached:
        return {"filename": file.filename, "from_cache": True, **cached}
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            print("[INFO] Starting parallel transcription and diarization...")
            
            transcribe_task = client.post(
                f"{WHISPER_URL}/transcribe",
                files={"file": (file.filename, file_content, file.content_type)},
            )
            
            diarize_task = client.post(
                f"{NEMO_URL}/diarize",
                files={"file": (file.filename, file_content, file.content_type)},
            )
            
            results = await asyncio.gather(
                transcribe_task,
                diarize_task,
                return_exceptions=True
            )
            
            transcribe_response, diarize_response = results
            
            if isinstance(transcribe_response, Exception):
                raise HTTPException(500, f"Transcription failed: {str(transcribe_response)}")
            
            if isinstance(diarize_response, Exception):
                raise HTTPException(500, f"Diarization failed: {str(diarize_response)}")
            
            if transcribe_response.status_code != 200:
                raise HTTPException(transcribe_response.status_code, f"Transcription failed: {transcribe_response.text}")
            
            if diarize_response.status_code != 200:
                raise HTTPException(diarize_response.status_code, f"Diarization failed: {diarize_response.text}")
            
            transcription = transcribe_response.json()
            diarization = diarize_response.json()
            
            print(f"[INFO] ✓ Transcription: {len(transcription.get('segments', []))} segments")
            print(f"[INFO] ✓ Diarization: {len(diarization.get('speakers', []))} speaker segments")
            
            formatted_transcript = format_diarized_transcript(transcription, diarization)
            
            speakers = diarization.get("speakers", [])
            unique_speakers = set(spk.get("speaker", "SPEAKER_00") for spk in speakers)
            num_speakers = len(unique_speakers)
            diarization["num_speakers"] = num_speakers
            
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
                except Exception as e:
                    print(f"[WARN] Summarization error: {e}")
                    summary = {"summary": "Summarization failed", "analysis": str(e)}
            
            raw_transcript = transcription.get("text", "")
            
            formatted_transcript_transliterated = None
            raw_transcript_transliterated = None
            
            if transliterate and TRANSLITERATION_AVAILABLE:
                print(f"[INFO] Generating transliterations...")
                formatted_transcript_transliterated = transliterate_transcript_with_speakers(formatted_transcript)
                raw_transcript_transliterated = transliterate_hindi_to_english(raw_transcript)
                print(f"[INFO] ✓ Transliteration complete")
            
            result = {
                "summary": summary,
                "formatted_transcript": formatted_transcript,
                "formatted_transcript_transliterated": formatted_transcript_transliterated,
                "raw_transcript": raw_transcript,
                "raw_transcript_transliterated": raw_transcript_transliterated,
                "transcription": transcription,
                "diarization": diarization,
                "num_speakers": num_speakers
            }
            
            file_cache.set(file_hash, result, len(file_content))
            
            print(f"[INFO] ✓ Full pipeline completed for {file.filename}")
            
            return {"filename": file.filename, "from_cache": False, **result}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline failed: {str(e)}")

# =============================================================================
# LIVE RECORDING ENDPOINTS
# =============================================================================
@app.post("/live/start")
async def live_start():
    global recording_thread, transcription_thread, recording_start_time
    global recording_duration_final, live_transcript, live_total_segments
    
    if not LIVE_RECORDING_AVAILABLE:
        raise HTTPException(503, "Live recording not available")
    
    if rec_state.is_recording:
        raise HTTPException(400, "Already recording")

    rec_state.reset()
    rec_state.start()

    recording_thread = threading.Thread(
        target=recording_worker,
        args=(rec_state, RECORDING_SAMPLE_RATE, INPUT_DEVICE_ID),
        daemon=True
    )
    transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(rec_state, RECORDING_SAMPLE_RATE, LIVE_CHUNK_DURATION),
        daemon=True
    )
    
    recording_thread.start()
    transcription_thread.start()

    recording_start_time = datetime.now()
    recording_duration_final = None
    live_transcript = ""
    live_total_segments = 0

    print("[INFO] Live recording started")
    
    return {
        "status": "recording_started",
        "chunk_duration": LIVE_CHUNK_DURATION,
        "sample_rate": RECORDING_SAMPLE_RATE
    }


@app.post("/live/stop")
async def live_stop():
    global recording_duration_final, recording_start_time

    if not rec_state.is_recording:
        raise HTTPException(400, "Not recording")

    rec_state.stop()

    if recording_start_time:
        recording_duration_final = (datetime.now() - recording_start_time).total_seconds()

    if transcription_thread and transcription_thread.is_alive():
        transcription_thread.join(timeout=10.0)

    print(f"[INFO] Recording stopped. Duration: {recording_duration_final:.2f}s")
    
    return {
        "status": "stopped",
        "duration": recording_duration_final,
        "total_segments": live_total_segments
    }


@app.get("/live/status", response_model=LiveStatusResponse)
async def live_status():
    global live_transcript, live_total_segments

    new_segments = rec_state.get_segments()
    latest_text = None
    
    # ADD SPEAKER LABEL (default to SPEAKER_0 for live without diarization)
    for seg in new_segments:
        # Format with speaker label
        speaker_label = "SPEAKER_0"  # Default for live transcription
        live_transcript += f"[{seg['timestamp']}] {speaker_label}: {seg['text']}\n"
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
async def live_diarize(
    generate_summary: bool = True,
    transliterate: bool = True
):
    """
    Diarize live recording with proper audio preprocessing
    """
    global live_transcript, recording_start_time, recording_duration_final
    
    if rec_state.is_recording:
        raise HTTPException(400, "Cannot diarize while recording is active")
    
    full_audio = rec_state.get_full_audio()
    if len(full_audio) == 0:
        raise HTTPException(400, "No audio recorded")
    
    logger.info(f"[DIARIZE] Starting diarization on {len(full_audio)} samples")
    
    # ========================================================================
    # CRITICAL FIX: Proper audio preprocessing for NeMo
    # ========================================================================
    
    def preprocess_audio_for_diarization(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio to ensure NeMo VAD doesn't detect it as silence
        """
        # 1. Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 2. Normalize to prevent clipping (important!)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Normalize to 70% of max range (not 100% to avoid clipping)
            audio_data = audio_data / max_val * 0.7
        
        # 3. Apply gentle high-pass filter to remove DC offset
        try:
            from scipy.signal import butter, lfilter
            
            # High-pass filter at 80 Hz
            nyquist = sample_rate / 2
            cutoff = 80 / nyquist
            
            if cutoff < 1.0:  # Valid cutoff
                b, a = butter(2, cutoff, btype='high')
                audio_data = lfilter(b, a, audio_data)
        except Exception as e:
            logger.warning(f"[DIARIZE] Filter failed: {e}, using raw audio")
        
        # 4. Apply soft amplification if audio is too quiet
        rms = np.sqrt(np.mean(audio_data ** 2))
        target_rms = 0.1  # Target RMS level
        
        if rms > 0 and rms < target_rms:
            gain = target_rms / rms
            # Limit gain to prevent over-amplification
            gain = min(gain, 3.0)
            audio_data = audio_data * gain
            logger.info(f"[DIARIZE] Applied gain: {gain:.2f}x (RMS {rms:.4f} → {target_rms:.4f})")
        
        # 5. Final normalization to int16 range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        logger.info(f"[DIARIZE] Audio preprocessed: RMS={rms:.4f}, Max={max_val:.4f}")
        
        return audio_int16
    
    # Preprocess audio
    processed_audio = preprocess_audio_for_diarization(full_audio, RECORDING_SAMPLE_RATE)
    
    # Save processed audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wavfile.write(tmp.name, RECORDING_SAMPLE_RATE, processed_audio)
        audio_path = tmp.name
    
    # Log audio stats for debugging
    audio_duration = len(processed_audio) / RECORDING_SAMPLE_RATE
    logger.info(f"[DIARIZE] Audio duration: {audio_duration:.2f}s, "
                f"Sample rate: {RECORDING_SAMPLE_RATE}Hz, "
                f"Samples: {len(processed_audio)}")
    
    try:
        # ====================================================================
        # STEP 1: Get existing transcript segments
        # ====================================================================
        existing_segments = rec_state.get_all_segments()
        
        if not existing_segments:
            raise HTTPException(400, "No transcript segments available")
        
        logger.info(f"[DIARIZE] Using {len(existing_segments)} existing segments")
        
        # ====================================================================
        # STEP 2: Perform diarization with relaxed VAD threshold
        # ====================================================================
        with open(audio_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            
            # Send with relaxed VAD parameters
            data = {
                'num_speakers': '',  # Auto-detect
                'vad_onset': '0.5',  # RELAXED: Was default 0.8 (lower = more sensitive)
                'vad_offset': '0.5',  # RELAXED: Was default 0.6
            }
            
            logger.info("[DIARIZE] Calling NeMo diarization service...")
            
            diarization_response = requests.post(
                f"{NEMO_URL}/diarize",
                files=files,
                data=data,
                timeout=300
            )
        
        if diarization_response.status_code != 200:
            error_detail = diarization_response.text
            logger.error(f"[DIARIZE] NeMo error: {error_detail}")
            
            # If diarization fails, fall back to single speaker
            logger.warning("[DIARIZE] Falling back to single-speaker mode")
            return fallback_single_speaker_diarization(existing_segments, generate_summary)
        
        diarization_result = diarization_response.json()
        speaker_segments = diarization_result.get('segments', [])
        
        if not speaker_segments:
            logger.warning("[DIARIZE] No speaker segments returned, using fallback")
            return fallback_single_speaker_diarization(existing_segments, generate_summary)
        
        logger.info(f"[DIARIZE] Got {len(speaker_segments)} speaker segments")
        
        # ====================================================================
        # STEP 3: Map existing transcripts to speakers
        # ====================================================================
        def timestamp_to_seconds(timestamp_str: str) -> float:
            """Convert HH:MM:SS to seconds"""
            try:
                parts = timestamp_str.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + float(s)
                elif len(parts) == 2:
                    m, s = parts
                    return int(m) * 60 + float(s)
                else:
                    return float(parts[0])
            except:
                return 0.0
        
        def assign_speaker_to_segment(seg_time: float, speaker_segments: list) -> str:
            """Assign speaker label based on time overlap"""
            for spk_seg in speaker_segments:
                start = spk_seg['start']
                end = spk_seg['end']
                speaker = spk_seg['speaker']
                
                if start <= seg_time <= end:
                    return speaker
            
            # Find closest speaker
            min_distance = float('inf')
            closest_speaker = "SPEAKER_0"
            
            for spk_seg in speaker_segments:
                start = spk_seg['start']
                end = spk_seg['end']
                speaker = spk_seg['speaker']
                
                if seg_time < start:
                    distance = start - seg_time
                elif seg_time > end:
                    distance = seg_time - end
                else:
                    distance = 0
                
                if distance < min_distance:
                    min_distance = distance
                    closest_speaker = speaker
            
            return closest_speaker
        
        # Map segments
        diarized_segments = []
        
        for seg in existing_segments:
            seg_time = timestamp_to_seconds(seg['timestamp'])
            speaker = assign_speaker_to_segment(seg_time, speaker_segments)
            
            diarized_segments.append({
                'timestamp': seg['timestamp'],
                'speaker': speaker,
                'text': seg['text'],
                'language': seg.get('language', 'hi')
            })
        
        logger.info(f"[DIARIZE] Mapped {len(diarized_segments)} segments to speakers")
        
        # ====================================================================
        # STEP 4: Format transcript
        # ====================================================================
        formatted_transcript = ""
        
        for seg in diarized_segments:
            if is_hallucination(seg['text']):
                continue
            
            formatted_transcript += f"[{seg['timestamp']}] {seg['speaker']}: {seg['text']}\n"
        
        # Normalize speaker labels
        formatted_transcript = normalize_speaker_labels(formatted_transcript)
        
        # ====================================================================
        # STEP 5: Generate summary
        # ====================================================================
        summary_result = None
        
        if generate_summary:
            try:
                logger.info("[DIARIZE] Generating MoM...")
                
                summary_response = requests.post(
                    f"{LLAMA_SUMMARIZER_URL}/summarize",
                    json={"text": formatted_transcript, "max_length": 4000, "temperature": 0.01},
                    timeout=300
                )
                
                if summary_response.status_code == 200:
                    summary_result = summary_response.json()
                    logger.info(f"[DIARIZE] MoM generated")
            except Exception as e:
                logger.error(f"[DIARIZE] Summary failed: {e}")
        
        # ====================================================================
        # Cleanup
        # ====================================================================
        os.unlink(audio_path)
        
        unique_speakers = len(set(re.findall(r'SPEAKER_\d+', formatted_transcript)))
        
        logger.info(f"[DIARIZE] ✓ Complete - {unique_speakers} speakers")
        
        return {
            "formatted_transcript": formatted_transcript,
            "num_speakers": unique_speakers,
            "summary": summary_result.get('analysis') if summary_result else None,
            "segments_count": len(diarized_segments)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DIARIZE] Failed: {e}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        
        raise HTTPException(500, f"Diarization failed: {str(e)}")




@app.post("/live/summarize")
async def live_summarize():
    global live_transcript
    
    if rec_state.is_recording:
        raise HTTPException(400, "Stop recording first")
    
    if not live_transcript:
        raise HTTPException(400, "No transcript available")
    
    try:
        lines = live_transcript.strip().split('\n')
        text_only = [
            line.split(':', 1)[1].strip() if ':' in line and ']' in line else line
            for line in lines
        ]
        combined_text = ' '.join(text_only)
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{LLAMA_URL}/summarize",
                json={"text": combined_text, "max_length": 600}
            )
            
            if response.status_code == 200:
                summary_data = response.json()
                return {
                    "status": "completed",
                    "summary": summary_data.get("analysis", summary_data.get("summary", ""))
                }
            else:
                raise HTTPException(response.status_code, response.text)
    
    except Exception as e:
        print(f"[ERROR] Summary failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/live/reset")
async def live_reset():
    global live_transcript, live_total_segments, recording_start_time, recording_duration_final
    
    if rec_state.is_recording:
        raise HTTPException(400, "Stop recording first")
    
    rec_state.reset()
    live_transcript = ""
    live_total_segments = 0
    recording_start_time = None
    recording_duration_final = None
    
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
        raise HTTPException(400, "No content")
    buffer = build_pdf_buffer(req.content, req.title or "Transcript")
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="transcript.pdf"'})


@app.post("/export/complete-txt")
async def export_complete_txt(req: CompleteExportRequest):
    """Export complete report as TXT - WITHOUT Raw Transcript section"""
    content_parts = [
        "=" * 80,
        "ANGELBOT.AI - COMPLETE MEETING REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"File: {req.filename}",
        f"Speakers: {req.speaker_count}",
        "=" * 80,
        "",
        "SECTION 1: DIARIZED TRANSCRIPT (Speaker-Labeled with Timestamps)",
        "-" * 80,
        req.formatted_transcript,
        "",
        "=" * 80,
        "SECTION 2: MINUTES OF MEETING",
        "-" * 80,
        req.summary if req.summary else "No Minutes of Meeting available",
        "",
        "=" * 80,
        "Generated by AngelBot.AI",
        "© 2026 AngelBot.AI · Whisper · NeMo · Llama · React",
        "=" * 80
    ]
    
    buffer = BytesIO("\n".join(content_parts).encode('utf-8'))
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{req.filename}_meeting_report.txt"'}
    )

@app.post("/export/complete-pdf")
async def export_complete_pdf(req: CompleteExportRequest):
    """
    Export professional meeting report as PDF
    MoM FIRST (starting from page 1), then Diarized Transcript
    """
    buffer = BytesIO()
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
        
        # Create PDF
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=f"Meeting Report - {req.filename}"
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # ============================================================
        # STYLES
        # ============================================================
        
        # Main title (MoM heading - centered)
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            spaceBefore=0,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=22
        )
        
        # Date/Time (right-aligned, smaller)
        datetime_style = ParagraphStyle(
            'DateTime',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=20,
            alignment=TA_RIGHT,
            fontName='Helvetica',
            leading=11
        )
        
        # Section headers (AGENDA, SUMMARY, SPEAKER-WISE NOTES, DECISIONS TAKEN, etc.)
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading1'],
            fontSize=12,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            leading=15,
            alignment=TA_LEFT
        )
        
        # MoM body text
        mom_body_style = ParagraphStyle(
            'MoMBody',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#334155'),
            fontName='Helvetica',
            leftIndent=0,
            spaceAfter=4,
            spaceBefore=2,
            alignment=TA_JUSTIFY
        )
        
        # MoM bullet/list items
        mom_list_style = ParagraphStyle(
            'MoMList',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#475569'),
            fontName='Helvetica',
            leftIndent=16,
            spaceAfter=3,
            bulletIndent=8,
            bulletFontName='Helvetica'
        )
        
        # Decision items (D1, D2, etc.) - better alignment
        decision_style = ParagraphStyle(
            'Decision',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#334155'),
            fontName='Helvetica',
            leftIndent=16,
            spaceAfter=5,
            bulletIndent=0,
            firstLineIndent=0
        )
        
        # Transcript section header
        transcript_section_style = ParagraphStyle(
            'TranscriptSection',
            parent=styles['Heading1'],
            fontSize=13,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=6,
            spaceBefore=24,
            fontName='Helvetica-Bold',
            leading=16,
            alignment=TA_LEFT
        )
        
        # Transcript body (monospace)
        transcript_style = ParagraphStyle(
            'TranscriptBody',
            parent=styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#1e293b'),
            fontName='Courier',
            leftIndent=0,
            rightIndent=0,
            spaceBefore=2,
            spaceAfter=2,
            wordWrap='LTR'
        )
        
        # Separator
        separator_style = ParagraphStyle(
            'Separator',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#cbd5e1'),
            fontName='Courier',
            alignment=TA_CENTER
        )
        
        # ============================================================
        # PAGE 1: MINUTES OF MEETING (STARTS IMMEDIATELY)
        # ============================================================
        
        # Title - Centered
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("MINUTES OF MEETING", title_style))
        
        # Date and Time - Right aligned, smaller
        datetime_text = f"Date: {datetime.now().strftime('%d %B %Y')}<br/>Time: {datetime.now().strftime('%I:%M %p')}"
        story.append(Paragraph(datetime_text, datetime_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Process MoM content
        if req.summary and req.summary.strip():
            lines = req.summary.split('\n')
            
            for line in lines:
                line = line.rstrip()
                
                # Skip empty lines but add spacing
                if not line.strip():
                    story.append(Spacer(1, 4))
                    continue
                
                # Escape special characters
                safe_line = (line.replace('&', '&amp;')
                               .replace('<', '&lt;')
                               .replace('>', '&gt;'))
                
                # Skip separator lines
                if all(c in '=_' for c in line.strip()):
                    story.append(Spacer(1, 6))
                    continue
                
                # Skip "MoM |" header line (already have title)
                if line.startswith('MoM |'):
                    continue
                
                # Skip Date/Time lines (already added)
                if line.startswith('Date:') or line.startswith('Time:'):
                    continue
                
                # Section headers (AGENDA, SUMMARY, ATTENDEES, SPEAKER-WISE NOTES, DECISIONS TAKEN, etc.)
                if line.strip().isupper() and len(line.strip()) > 3:
                    # Special handling for common section names
                    if any(keyword in line.strip() for keyword in ['AGENDA', 'SUMMARY', 'ATTENDEES', 'SPEAKER', 'DECISION', 'ACTION', 'PURPOSE']):
                        story.append(Paragraph(safe_line, section_header_style))
                    else:
                        story.append(Paragraph(safe_line, section_header_style))
                
                # Decision items (D1, D2, etc.) - improved formatting
                elif re.match(r'^D\d+\.', line.strip()):
                    story.append(Paragraph(safe_line, decision_style))
                
                # Bullet points
                elif line.strip().startswith('•') or line.strip().startswith('-'):
                    story.append(Paragraph(safe_line, mom_list_style))
                
                # Skip "Regards," line
                elif line.strip().startswith('Regards'):
                    continue
                
                # Skip footer lines (Generated by AngelBot, copyright, etc.)
                elif any(keyword in line for keyword in ['Generated by AngelBot', 'AngelBot.AI', '© 2026', 'Powered by']):
                    continue
                
                # Indented content
                elif line.startswith('  ') or line.startswith('    '):
                    indent_level = len(line) - len(line.lstrip())
                    custom_style = ParagraphStyle(
                        'IndentedText',
                        parent=mom_body_style,
                        leftIndent=indent_level * 3
                    )
                    story.append(Paragraph(safe_line, custom_style))
                
                # Action item details
                elif ':' in line and any(kw in line for kw in ['Owner:', 'Due Date:', 'Deadline:']):
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{safe_line}", mom_list_style))
                
                # Regular body text
                else:
                    story.append(Paragraph(safe_line, mom_body_style))
        
        else:
            story.append(Paragraph("No Minutes of Meeting generated.", mom_body_style))
        
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("=" * 90, separator_style))
        story.append(Spacer(1, 0.4*inch))
        
        # ============================================================
        # SECTION 2: DIARIZED TRANSCRIPT
        # ============================================================
        annexure_title_style = ParagraphStyle(
            'AnnexureTitle',
            parent=styles['Heading1'],
            fontSize=13,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=6,
            spaceBefore=24,
            fontName='Helvetica-Bold',
            leading=16,
            alignment=TA_CENTER  # ← CENTERED
        )

        story.append(Paragraph("ANNEXURE", annexure_title_style))
        story.append(Paragraph("<i>Speaker-Labeled with Timestamps</i>", 
                              ParagraphStyle('SubHeader', parent=styles['Normal'], 
                                           fontSize=9, textColor=colors.HexColor('#64748b'),
                                           fontName='Helvetica-Oblique', spaceAfter=12)))
        story.append(Spacer(1, 0.1*inch))
        
        # Process transcript
        for line in req.formatted_transcript.split('\n'):
            if line.strip():
                safe_line = (line.replace('&', '&amp;')
                               .replace('<', '&lt;')
                               .replace('>', '&gt;'))
                story.append(Paragraph(safe_line, transcript_style))
            else:
                story.append(Spacer(1, 3))
        
        # Build PDF (NO FOOTER)
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{req.filename}_meeting_report.pdf"'
            }
        )
        
    except Exception as e:
        logger.error(f"[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)