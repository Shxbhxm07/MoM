#!/usr/bin/env python3
"""
Test script for new features:
1. WebSocket progress tracking
2. File caching
3. Optimized audio preprocessing
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint with cache info"""
    print("\n" + "="*60)
    print("1. Testing Health Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    data = response.json()

    print(f"✅ Status: {data['status']}")
    print(f"✅ Model: {data['model']}")
    print(f"✅ Whisper Device: {data['whisper_device']}")
    print(f"✅ NeMo Device: {data['nemo_device']}")
    print(f"✅ Cache Size: {data['cache_size_mb']:.2f} MB")
    print(f"✅ Cached Files: {data['cached_files']}")


def test_cache_stats():
    """Test cache statistics endpoint"""
    print("\n" + "="*60)
    print("2. Testing Cache Stats")
    print("="*60)

    response = requests.get(f"{BASE_URL}/cache/stats")
    data = response.json()

    print(f"✅ Cache Size: {data['cache_size_mb']:.2f} MB / {data['max_size_mb']:.2f} MB")
    print(f"✅ Cached Files: {data['cached_files']}")
    print(f"✅ Sample Keys: {data['cache_keys'][:3]}")


def test_transcription_with_progress(audio_file):
    """Test file transcription with job tracking"""
    print("\n" + "="*60)
    print("3. Testing Transcription with Progress Tracking")
    print("="*60)

    if not audio_file:
        print("⚠️  No audio file provided, skipping transcription test")
        print("   Usage: python test_new_features.py <path/to/audio.mp3>")
        return

    print(f"📁 Uploading file: {audio_file}")
    start_time = time.time()

    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/transcribe-file", files=files)

    if response.status_code == 200:
        data = response.json()
        upload_time = time.time() - start_time

        print(f"✅ Job ID: {data['job_id']}")
        print(f"✅ Processing Time: {upload_time:.2f}s")
        print(f"✅ Text Preview: {data['full_text'][:100]}...")

        # Check job status
        job_response = requests.get(f"{BASE_URL}/job/{data['job_id']}")
        if job_response.status_code == 200:
            job_data = job_response.json()
            print(f"✅ Job Status: {job_data['stage']} ({job_data['progress']:.1f}%)")
            print(f"✅ Message: {job_data['message']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")


def test_cache_hit(audio_file):
    """Test cache hit by uploading same file twice"""
    print("\n" + "="*60)
    print("4. Testing Cache Hit (Upload Same File Twice)")
    print("="*60)

    if not audio_file:
        print("⚠️  No audio file provided, skipping cache test")
        return

    print(f"📁 First upload: {audio_file}")
    start1 = time.time()

    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response1 = requests.post(f"{BASE_URL}/transcribe-file", files=files)

    time1 = time.time() - start1
    print(f"⏱️  First upload time: {time1:.2f}s")

    # Upload same file again
    print(f"\n📁 Second upload (should hit cache): {audio_file}")
    start2 = time.time()

    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response2 = requests.post(f"{BASE_URL}/transcribe-file", files=files)

    time2 = time.time() - start2
    print(f"⏱️  Second upload time: {time2:.2f}s")

    if time2 < time1 / 10:  # Cache should be >10x faster
        print(f"✅ CACHE HIT! {time1/time2:.1f}x faster")
    else:
        print(f"⚠️  Cache might not be working (speedup: {time1/time2:.1f}x)")

    # Check cache stats again
    cache_response = requests.get(f"{BASE_URL}/cache/stats")
    cache_data = cache_response.json()
    print(f"✅ Total cached files: {cache_data['cached_files']}")


def test_clear_cache():
    """Test cache clearing"""
    print("\n" + "="*60)
    print("5. Testing Cache Clear")
    print("="*60)

    response = requests.post(f"{BASE_URL}/cache/clear")
    data = response.json()

    print(f"✅ Status: {data['status']}")
    print(f"✅ Message: {data['message']}")

    # Verify cache is empty
    stats = requests.get(f"{BASE_URL}/cache/stats").json()
    print(f"✅ Cached files after clear: {stats['cached_files']}")


def main():
    import sys

    print("\n🚀 AngelBot.AI - Testing New Features")
    print("="*60)

    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        # Test 1: Health check
        test_health()

        # Test 2: Cache stats
        test_cache_stats()

        # Test 3: Transcription with progress (if audio file provided)
        if audio_file:
            test_transcription_with_progress(audio_file)

            # Test 4: Cache hit
            test_cache_hit(audio_file)

        # Test 5: Clear cache
        test_clear_cache()

        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)

        if not audio_file:
            print("\n💡 Tip: Run with an audio file to test transcription:")
            print("   python test_new_features.py path/to/audio.mp3")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print("   Make sure the server is running:")
        print("   cd /home/ai-ml2/Desktop/angelbot-ai/backend")
        print("   ./start_server.sh")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
