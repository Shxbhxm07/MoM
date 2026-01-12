FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    torch

WORKDIR /app
COPY service.py /app/

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
