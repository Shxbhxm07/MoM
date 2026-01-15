from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
from typing import Optional
from pathlib import Path
import tempfile
import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import NeMo (optional)
NEMO_AVAILABLE = False
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
    logger.info("✓ NeMo is available")
except ImportError as e:
    logger.warning(f"NeMo not available: {e}")
    logger.info("Service will run with Whisper only")

app = FastAPI(
    title="Unified Speech Service",
    description="Whisper transcription" + (" + NeMo ASR" if NEMO_AVAILABLE else ""),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "whisper": None,
    "nemo_asr": None,
}

loading_status = {
    "whisper": "not_started",
    "nemo": "not_available" if not NEMO_AVAILABLE else "not_started"
}

# Track which Whisper model is loaded
CURRENT_WHISPER_MODEL = None


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global CURRENT_WHISPER_MODEL
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("No GPU detected")
        
        # Load Whisper - try different models in order of preference
        logger.info(f"Loading Whisper on {device}...")
        loading_status["whisper"] = "loading"
        
        # Try to load models in order: large-v3, large-v2, large, base
        model_priority = ["large-v3", "large-v2", "large", "base"]
        loaded = False
        
        for model_name in model_priority:
            try:
                logger.info(f"Attempting to load {model_name}...")
                models["whisper"] = whisper.load_model(
                    model_name, 
                    device=device,
                    download_root="/root/.cache/whisper"  # Use cache directory
                )
                CURRENT_WHISPER_MODEL = model_name
                loading_status["whisper"] = f"loaded ({model_name})"
                logger.info(f"✓ Whisper {model_name} loaded successfully!")
                loaded = True
                break
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
                continue
        
        if not loaded:
            logger.error("Failed to load any Whisper model")
            loading_status["whisper"] = "failed"
            raise Exception("No Whisper model could be loaded")
        
        # Load NeMo if available
        if NEMO_AVAILABLE:
            try:
                logger.info("Loading NeMo ASR...")
                loading_status["nemo"] = "loading"
                models["nemo_asr"] = nemo_asr.models.EncDecCTCModel.from_pretrained(
                    "stt_en_conformer_ctc_small"
                )
                loading_status["nemo"] = "loaded"
                logger.info("✓ NeMo loaded")
            except Exception as e:
                logger.error(f"Failed to load NeMo: {e}")
                loading_status["nemo"] = "failed"
        
        logger.info("🚀 Service ready!")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    return {
        "service": "Unified Speech Service",
        "version": "1.0.0",
        "whisper_model": CURRENT_WHISPER_MODEL,
        "models_available": {
            "whisper": models["whisper"] is not None,
            "nemo": NEMO_AVAILABLE and models["nemo_asr"] is not None
        },
        "endpoints": {
            "/health": "Health check",
            "/transcribe": "Whisper transcription (POST)",
            "/asr": "NeMo ASR or Whisper (POST)",
            "/full-pipeline": "Both models (POST)" if NEMO_AVAILABLE else "Same as /transcribe"
        }
    }


@app.get("/health")
async def health():
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
        }
    
    return {
        "status": "healthy" if models["whisper"] else "degraded",
        "whisper_model": CURRENT_WHISPER_MODEL,
        "loading_status": loading_status,
        "models_loaded": {
            "whisper": models["whisper"] is not None,
            "nemo_asr": models["nemo_asr"] is not None
        },
        "nemo_available": NEMO_AVAILABLE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    beam_size: Optional[int] = Form(5),
    best_of: Optional[int] = Form(5),
    word_timestamps: Optional[str] = Form('false'),
    task: Optional[str] = Form('transcribe'),
    temperature: Optional[float] = Form(0.0),  # ADD THIS
    condition_on_previous_text: Optional[str] = Form('false'),  # ADD THIS
    compression_ratio_threshold: Optional[float] = Form(2.4),  # ADD THIS
    logprob_threshold: Optional[float] = Form(-1.0),  # ADD THIS
    no_speech_threshold: Optional[float] = Form(0.6),  # ADD THIS
):
    """Transcribe with your exact Whisper configuration"""
    if not models["whisper"]:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        enable_word_timestamps = word_timestamps.lower() == 'true'
        condition_prev = condition_on_previous_text.lower() == 'true'

        logger.info(f"Transcribing: {file.filename} (lang={language}, temp={temperature})")

        # Prepare transcribe options with your settings
        transcribe_options = {
            "fp16": torch.cuda.is_available(),
            "word_timestamps": enable_word_timestamps,
            "task": task,
            "temperature": temperature,
            "condition_on_previous_text": condition_prev,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }
        
        if language:
            transcribe_options["language"] = language
        if beam_size is not None:
            transcribe_options["beam_size"] = beam_size
        if best_of is not None:
            transcribe_options["best_of"] = best_of

        # Transcribe
        result = models["whisper"].transcribe(temp_file, **transcribe_options)

        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown")
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    """NeMo ASR or fallback to Whisper"""
    if NEMO_AVAILABLE and models["nemo_asr"]:
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                temp_file = tmp.name
            
            logger.info(f"NeMo ASR: {file.filename}")
            transcription = models["nemo_asr"].transcribe([temp_file])[0]
            return {"text": transcription, "model": "nemo"}
            
        except Exception as e:
            logger.error(f"NeMo error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    else:
        # Fallback to Whisper
        logger.info("Using Whisper (NeMo not available)")
        result = await transcribe_audio(file)
        result["model"] = "whisper"
        return result


@app.post("/full-pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    use_whisper: bool = True,
    use_nemo: bool = False
):
    """Combined pipeline"""
    results = {}
    temp_file = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        # Whisper
        if use_whisper and models["whisper"]:
            logger.info("Running Whisper...")
            whisper_result = models["whisper"].transcribe(
                temp_file,
                fp16=torch.cuda.is_available()
            )
            results["whisper"] = {
                "text": whisper_result["text"],
                "language": whisper_result.get("language"),
                "segments": whisper_result.get("segments", [])
            }
        
        # NeMo
        if use_nemo and NEMO_AVAILABLE and models["nemo_asr"]:
            logger.info("Running NeMo...")
            nemo_result = models["nemo_asr"].transcribe([temp_file])[0]
            results["nemo"] = {"text": nemo_result}
        elif use_nemo and not NEMO_AVAILABLE:
            results["nemo"] = {"error": "NeMo not available in this container"}
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")


@app.get("/models")
async def list_models():
    return {
        "current_whisper_model": CURRENT_WHISPER_MODEL,
        "available_models": {
            "whisper": {
                "installed": True,
                "loaded": models["whisper"] is not None,
                "status": loading_status["whisper"],
                "model_name": CURRENT_WHISPER_MODEL
            },
            "nemo_asr": {
                "installed": NEMO_AVAILABLE,
                "loaded": models["nemo_asr"] is not None,
                "status": loading_status["nemo"]
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)