#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         FINDING AI MODELS IN YOUR SYSTEM                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to check common model locations
check_models() {
    local container=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Checking: $container"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Common model directories
    dirs=(
        "/app/models"
        "/models"
        "/root/.cache"
        "/root/.local"
        "~/.cache"
        "/opt/models"
        "/usr/local/models"
        "/app/.cache"
        "/tmp/models"
    )
    
    for dir in "${dirs[@]}"; do
        result=$(docker exec $container find $dir -type f \( -name "*.bin" -o -name "*.pth" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pt" \) 2>/dev/null | head -5)
        if [ -n "$result" ]; then
            echo "✓ Found models in: $dir"
            echo "$result"
            echo ""
        fi
    done
    
    # Check working directory
    echo "Working directory contents:"
    docker exec $container ls -lh /app 2>/dev/null | grep -E "model|weight|checkpoint" || echo "  No obvious model files in /app"
    echo ""
}

# Check each service
check_models "speech-unified"
check_models "llama-summarizer"
check_models "nemo-diarizer"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECKING DOCKER VOLUMES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check volumes
docker volume ls | grep -E "model|cache|huggingface"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECKING HOST MACHINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check host for common model locations
echo "Common locations on host:"
find ~/kangelbot-ai -type f \( -name "*.bin" -o -name "*.pth" -o -name "*.onnx" \) 2>/dev/null | head -10 || echo "  No models found in project directory"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         CHECKING DOCKER COMPOSE CONFIGURATION                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ -f "docker-compose.yml" ]; then
    echo "Volume mounts in docker-compose.yml:"
    grep -A 5 "volumes:" docker-compose.yml | grep -v "^--$"
else
    echo "docker-compose.yml not found in current directory"
fi
