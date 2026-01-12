# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import logging
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Llama 3 Summarization Service", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# tokenizer = None
# model = None

# # Maximum safe input tokens (leave room for prompt + output)
# MAX_INPUT_TOKENS = 6000  # Conservative limit for 8K context window

# class SummarizeRequest(BaseModel):
#     text: str
#     max_length: int = 300
#     temperature: float = 0.7

# def chunk_text(text: str, max_tokens: int) -> list:
#     """Split text into chunks that fit within token limit"""
#     # Rough estimate: 1 token ≈ 4 characters
#     max_chars = max_tokens * 4
    
#     # Split by sentences for better coherence
#     sentences = text.split('. ')
    
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for sentence in sentences:
#         sentence_length = len(sentence)
        
#         if current_length + sentence_length > max_chars:
#             if current_chunk:
#                 chunks.append('. '.join(current_chunk) + '.')
#             current_chunk = [sentence]
#             current_length = sentence_length
#         else:
#             current_chunk.append(sentence)
#             current_length += sentence_length
    
#     if current_chunk:
#         chunks.append('. '.join(current_chunk) + '.')
    
#     return chunks

# @app.on_event("startup")
# async def load_model():
#     global tokenizer, model
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         if torch.cuda.is_available():
#             logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
#             logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
#         model_path = "/app/Meta-Llama-3-8B-Instruct"
#         logger.info(f"Loading Llama 3 8B from: {model_path}")
        
#         tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto",
#             local_files_only=True
#         )
        
#         logger.info(f"✓ Llama 3 loaded on {device}")
#         logger.info("🚀 Summarization service ready!")
        
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")

# @app.get("/")
# async def root():
#     return {
#         "service": "Llama 3 Summarization",
#         "model": "Meta-Llama-3-8B-Instruct",
#         "endpoints": {"/health": "Health check", "/summarize": "Summarize text (POST)"}
#     }

# @app.get("/health")
# async def health():
#     return {
#         "status": "healthy" if model else "loading",
#         "model_loaded": model is not None,
#         "gpu_available": torch.cuda.is_available()
#     }

# def generate_summary_for_chunk(chunk: str, max_length: int, temperature: float) -> str:
#     """Generate summary for a single chunk"""
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Provide only the summary."},
#         {"role": "user", "content": f"Summarize this in 2-3 sentences:\n\n{chunk}"}
#     ]
    
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
#     input_length = inputs.input_ids.shape[1]
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_length,
#             temperature=temperature,
#             do_sample=True,
#             top_p=0.9,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     generated_ids = outputs[0][input_length:]
#     summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
#     # Cleanup
#     summary = summary.replace("<|eot_id|>", "").strip()
#     if "assistant" in summary.lower():
#         summary = summary.split("assistant")[-1].strip()
    
#     return summary

# @app.post("/summarize")
# async def summarize(request: SummarizeRequest):
#     """Summarize text with automatic chunking for long inputs"""
#     if not model or not tokenizer:
#         raise HTTPException(503, "Model not loaded")
    
#     try:
#         text_length = len(request.text)
#         logger.info(f"Summarizing text ({text_length} chars)")
        
#         # Estimate tokens (rough: 4 chars = 1 token)
#         estimated_tokens = text_length // 4
        
#         # If text is short enough, process directly
#         if estimated_tokens < MAX_INPUT_TOKENS:
#             logger.info("Processing in single pass")
#             summary = generate_summary_for_chunk(request.text, request.max_length, request.temperature)
#         else:
#             # Chunk and summarize
#             logger.info(f"Text too long ({estimated_tokens} tokens). Chunking...")
#             chunks = chunk_text(request.text, MAX_INPUT_TOKENS)
#             logger.info(f"Split into {len(chunks)} chunks")
            
#             # Summarize each chunk
#             chunk_summaries = []
#             for i, chunk in enumerate(chunks):
#                 logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
#                 chunk_summary = generate_summary_for_chunk(chunk, 150, request.temperature)
#                 chunk_summaries.append(chunk_summary)
            
#             # Combine chunk summaries
#             combined_summary = " ".join(chunk_summaries)
            
#             # If combined summary is still long, summarize it again
#             if len(combined_summary) > 2000:
#                 logger.info("Final summary pass...")
#                 summary = generate_summary_for_chunk(combined_summary, request.max_length, request.temperature)
#             else:
#                 summary = combined_summary
        
#         logger.info(f"✓ Summary generated ({len(summary)} chars)")
        
#         return {
#             "summary": summary,
#             "original_length": text_length,
#             "summary_length": len(summary),
#             "compression_ratio": f"{len(summary) / text_length * 100:.1f}%"
#         }
        
#     except Exception as e:
#         logger.error(f"Summarization error: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise HTTPException(500, str(e))


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama 3 Meeting Intelligence Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None

# Conservative limit for Llama-3 8B (8K context)
MAX_INPUT_TOKENS = 6000

# =========================
# UNIVERSAL MEETING PROMPT
# =========================
MEETING_ANALYSIS_SYSTEM_PROMPT = """
You are Docutalk, an AI language model developed by Appolo Computers.
You are a specialized assistant for analyzing and structuring information
from audio transcriptions of meetings, discussions, or conversations.

The input text is a transcription of spoken audio.
It may be a complete transcription or a partial segment of a longer recording.

The transcription may contain informal language, repetitions, disfluencies,
incomplete sentences, multiple speakers, or mid-sentence starts and endings.

Your task is to analyze ONLY the provided transcription and produce a
clear, structured output with the following sections:

1. Purpose of the Discussion
   - State the main reason for the discussion.
   - If not explicitly stated, infer conservatively.
   - If unclear, write: "Not clearly stated in this segment."

2. Key Discussion Points
   - Summarize the main topics discussed.
   - Preserve logical or chronological flow.
   - Exclude filler words and transcription artifacts.

3. Decisions Made
   - List ONLY decisions that are explicitly stated or clearly agreed upon.
   - Do NOT infer decisions from personal opinions or goals.
   - If none are present, write: "No explicit decisions mentioned."

4. Action Items
   - List ONLY tasks that are explicitly committed to in the transcription.
   - An action item MUST include clear commitment language such as:
     "I will", "we will", "let’s do", "you take care of", "assigned to".
   - Do NOT convert personal goals, wishes, resolutions, or ideas
     into action items.
   - Include responsible person and timeline ONLY if explicitly stated.
   - If no explicit action items are present, write:
     "No explicit action items mentioned."

Rules:
- Base your response strictly on the provided transcription.
- Do NOT invent, swap, or assume responsibilities.
- Do NOT infer intent beyond what is explicitly stated.
- If the transcription is partial, summarize only what is present.
- Maintain clarity, accuracy, and professional tone.
""".strip()

# =========================
# REQUEST MODEL
# =========================
class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 600
    temperature: float = 0.2

# =========================
# CHUNKING LOGIC
# =========================
def chunk_text(text: str, max_tokens: int) -> list:
    """
    Rough chunking using char approximation (1 token ≈ 4 chars)
    Sentence-aware to preserve meaning.
    """
    max_chars = max_tokens * 4
    sentences = text.split(". ")

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_chars:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

# =========================
# MODEL LOADING
# =========================
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        model_path = "/app/Meta-Llama-3-8B-Instruct"
        logger.info(f"Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            local_files_only=True,
        )

        logger.info(f"✓ Model loaded on {device}")
        logger.info("🚀 Meeting Intelligence service ready")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise

# =========================
# HEALTH ENDPOINTS
# =========================
@app.get("/")
async def root():
    return {
        "service": "Llama 3 Meeting Intelligence",
        "model": "Meta-Llama-3-8B-Instruct",
        "endpoint": "/summarize",
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "loading",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
    }

# =========================
# CORE GENERATION FUNCTION
# =========================
def generate_analysis_for_chunk(
    chunk: str, max_length: int, temperature: float
) -> str:
    messages = [
        {"role": "system", "content": MEETING_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Audio Transcription:\n{chunk}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    result = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Cleanup safety
    result = result.replace("<|eot_id|>", "").strip()
    if "assistant" in result.lower():
        result = result.split("assistant")[-1].strip()

    return result

# =========================
# MAIN ENDPOINT
# =========================
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    if not model or not tokenizer:
        raise HTTPException(503, "Model not loaded")

    try:
        text_len = len(request.text)
        logger.info(f"Processing transcription ({text_len} chars)")

        estimated_tokens = text_len // 4

        # ---- SHORT TRANSCRIPTION ----
        if estimated_tokens < MAX_INPUT_TOKENS:
            logger.info("Single-pass analysis")
            analysis = generate_analysis_for_chunk(
                request.text,
                request.max_length,
                request.temperature,
            )

        # ---- LONG TRANSCRIPTION ----
        else:
            logger.info("Chunking long transcription")
            chunks = chunk_text(request.text, MAX_INPUT_TOKENS)
            logger.info(f"Total chunks: {len(chunks)}")

            chunk_results = []
            for idx, chunk in enumerate(chunks):
                logger.info(f"Analyzing chunk {idx + 1}/{len(chunks)}")
                chunk_result = generate_analysis_for_chunk(
                    chunk, max_length=400, temperature=request.temperature
                )
                chunk_results.append(chunk_result)

            combined = "\n\n".join(chunk_results)

            # Final merge pass
            if len(combined) > 2000:
                logger.info("Final merge analysis")
                analysis = generate_analysis_for_chunk(
                    combined,
                    request.max_length,
                    request.temperature,
                )
            else:
                analysis = combined

        return {
            "analysis": analysis,
            "original_length": text_len,
            "analysis_length": len(analysis),
            "compression_ratio": f"{len(analysis) / text_len * 100:.1f}%",
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(500, str(e))

