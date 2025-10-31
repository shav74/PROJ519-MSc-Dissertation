from pathlib import Path
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess

# Use moviepy's bundled ffmpeg
ffmpeg_path = get_ffmpeg_exe()
print(f"Using FFmpeg: {ffmpeg_path}\n")

video_dir = Path(r"..\..\data\test_videos")
output_base = Path(r"..\..\data\images_eval\own")

video_files = list(video_dir.glob("*.mp4"))

for video_file in video_files:
    video_name = video_file.stem
    output_dir = output_base / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / "frame_%06d.jpg")

    print(f"Processing: {video_file.name}")

    cmd = [
        ffmpeg_path,  # Use bundled ffmpeg
        '-y',
        '-i', str(video_file),
        '-vf', 'fps=1',
        output_pattern
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        frame_count = len(list(output_dir.glob("frame_*.jpg")))
        print(f"✓ Extracted {frame_count} frames\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr[-300:]}\n")

print("=== Done! ===")
