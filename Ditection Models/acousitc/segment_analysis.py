import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load("data/live/long shot.wav", sr=48000, mono=True)

# Define your flight segments (adjust these times based on your video)
segments = {
    "Background": (3, 5),      # First 10 seconds before takeoff
    "Hover": (5, 9),          # 20 seconds hovering
    "Approach": (9, 16),       # Approaching camera
    "Fly-by": (16, 23)         # Lateral pass
}

print("Segment Analysis")
segment_results = {}

for segment_name, (start, end) in segments.items():
    segment_audio = audio[start*sr:end*sr]
    rms = np.sqrt(np.mean(segment_audio**2))
    db_level = 20 * np.log10(rms + 1e-10) + 92
    segment_results[segment_name] = db_level
    print(f"{segment_name:12s}: {db_level:6.2f} dB")

# Calculate drone noise above background
drone_noise = {}
background_level = segment_results["Background"]

for segment_name, level in segment_results.items():
    if segment_name != "Background":
        difference = level - background_level
        drone_noise[segment_name] = difference
        print(f"\n{segment_name} above background: +{difference:.2f} dB")

# Plot
plt.figure(figsize=(10, 6))
segments_list = list(segment_results.keys())
db_values = list(segment_results.values())
plt.bar(segments_list, db_values, color=['gray', 'blue', 'orange', 'green'])
plt.ylabel('Relative dB Level (dBFS)')
plt.title('Acoustic Levels by Flight Segment')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('segment_analysis.png', dpi=300)
plt.show()

print("\nPlot saved as 'segment_analysis.png'")
