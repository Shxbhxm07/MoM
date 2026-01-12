from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
import tempfile
import os
import json
import shutil
from omegaconf import OmegaConf
from contextlib import asynccontextmanager
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_cfg = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global base_cfg
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        logger.info("Loading NeMo components...")
        
        # Create persistent output directory
        os.makedirs("/tmp/nemo_outputs", exist_ok=True)
        os.makedirs("/tmp/nemo_outputs/speaker_outputs", exist_ok=True)
        os.makedirs("/tmp/nemo_outputs/speaker_outputs/embeddings", exist_ok=True)
        os.makedirs("/tmp/nemo_outputs/pred_rttms", exist_ok=True)
        
        base_cfg = {
            'device': device,
            'num_workers': 0,
            'sample_rate': 16000,
            'batch_size': 64,
            'verbose': True,
            'diarizer': {
                'manifest_filepath': None,
                'out_dir': '/tmp/nemo_outputs',
                'oracle_vad': False,
                'collar': 0.25,
                'ignore_overlap': True,
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': [1.5, 0.75],
                        'shift_length_in_sec': [0.75, 0.375],
                        'multiscale_weights': [1, 1],
                        'save_embeddings': True  # CRITICAL: Must save embeddings
                    }
                },
                'msdd_model': {
                    'model_path': 'diar_msdd_telephonic',
                    'parameters': {
                        'use_speaker_model_from_ckpt': True,
                        'infer_batch_size': 25,
                        'sigmoid_threshold': [0.7],
                        'seq_eval_mode': False,
                        'split_infer': False,  # Changed to False to avoid issues
                        'diar_window_length': 50,
                        'overlap_infer_spk_limit': 5
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': False,
                        'max_num_speakers': 10,
                        'enhanced_count_thres': 80,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30
                    }
                },
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.63,
                        'shift_length_in_sec': 0.02,
                        'smoothing': False,
                        'overlap': 0.5,
                        'onset': 0.8,
                        'offset': 0.5,
                        'pad_onset': 0,
                        'pad_offset': 0,
                        'min_duration_on': 0,
                        'min_duration_off': 0.6,
                        'filter_speech_first': True
                    }
                }
            }
        }
        
        logger.info("✓ NeMo configuration loaded!")
        logger.info("🚀 Diarization service ready!")
        
    except Exception as e:
        logger.error(f"Error loading NeMo: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(
    title="NeMo Diarization Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_wav(input_path, output_path):
    """Convert audio to mono 16kHz WAV using ffmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        return True
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

@app.get("/")
async def root():
    return {
        "service": "NeMo Diarization",
        "status": "ready",
        "endpoints": {
            "/health": "Health check",
            "/diarize": "Speaker diarization (POST)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "nemo_available": base_cfg is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    if not base_cfg:
        raise HTTPException(503, "NeMo not loaded")
    
    temp_dir = None
    
    try:
        from nemo.collections.asr.models.msdd_models import ClusteringDiarizer
        
        logger.info(f"Diarizing: {file.filename}")
        
        # Create unique temp directory for this request
        temp_dir = tempfile.mkdtemp(prefix="nemo_diarize_")
        
        # Create all necessary subdirectories
        os.makedirs(os.path.join(temp_dir, "speaker_outputs"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "speaker_outputs", "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "pred_rttms"), exist_ok=True)
        
        # Save uploaded file
        upload_path = os.path.join(temp_dir, 'upload' + os.path.splitext(file.filename)[1])
        with open(upload_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Convert to proper WAV format
        audio_path = os.path.join(temp_dir, 'audio.wav')
        logger.info("Converting audio to mono 16kHz WAV...")
        if not convert_to_wav(upload_path, audio_path):
            raise HTTPException(400, "Failed to convert audio file")
        
        # Create manifest
        manifest = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }
        
        manifest_path = os.path.join(temp_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            f.write(json.dumps(manifest) + '\n')
        
        # Create config with this request's temp directory
        cfg = OmegaConf.create(base_cfg)
        cfg.diarizer.manifest_filepath = manifest_path
        cfg.diarizer.out_dir = temp_dir
        
        # Initialize and run diarizer
        logger.info("Initializing NeuralDiarizer...")
        diarizer = ClusteringDiarizer(cfg=cfg)
        
        logger.info("Running diarization...")
        diarizer.diarize()
        
        # Read results
        rttm_file = os.path.join(temp_dir, 'pred_rttms', 'audio.rttm')
        
        speakers = []
        if os.path.exists(rttm_file):
            logger.info(f"Reading RTTM file: {rttm_file}")
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        speakers.append({
                            "speaker": parts[7],
                            "start": float(parts[3]),
                            "duration": float(parts[4]),
                            "end": float(parts[3]) + float(parts[4])
                        })
        else:
            logger.error(f"RTTM file not found: {rttm_file}")
            logger.error(f"Directory contents: {os.listdir(temp_dir)}")
            if os.path.exists(os.path.join(temp_dir, 'pred_rttms')):
                logger.error(f"pred_rttms contents: {os.listdir(os.path.join(temp_dir, 'pred_rttms'))}")
        
        num_speakers = len(set([s['speaker'] for s in speakers]))
        logger.info(f"✓ Found {num_speakers} speakers with {len(speakers)} segments")
        
        return {
            "filename": file.filename,
            "speakers": speakers,
            "num_speakers": num_speakers
        }
        
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))
    
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)