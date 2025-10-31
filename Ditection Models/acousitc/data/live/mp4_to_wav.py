import subprocess
import os

# Input and output paths
input_video = "1.mp4"
output_audio = "1.wav"

# Extract audio using FFmpeg
subprocess.run([
    'ffmpeg', '-i', input_video,
    '-vn',  # No video
    '-acodec', 'pcm_s16le',  # PCM 16-bit
    '-ar', '48000',  # 48kHz sample rate
    '-ac', '1',  # Mono
    output_audio
])

print(f"Audio extracted to {output_audio}")
