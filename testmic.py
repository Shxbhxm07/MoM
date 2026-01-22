def get_audio_device_id():
    """Auto-detect EMEET microphone"""
    import sounddevice as sd
    import os
    
    # Check environment variable first
    env_device = os.getenv('AUDIO_DEVICE_ID')
    if env_device:
        try:
            return int(env_device)
        except:
            pass
    
    # Auto-detect EMEET
    for idx, device in enumerate(sd.query_devices()):
        if 'EMEET' in device['name'] and device['max_input_channels'] > 0:
            print(f"[INFO] Found EMEET at device {idx}")
            return idx
    
    return sd.default.device[0]

# Use it:
input_device = get_audio_device_id()