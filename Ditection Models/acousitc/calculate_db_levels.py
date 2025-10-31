import numpy as np
import librosa
from scipy.signal import firwin, lfilter

# Load audio
audio, sr = librosa.load("data/live/long shot.wav", sr=48000, mono=True)

# Apply A-weighting filter (approximate)
nyquist = sr / 2
frequencies = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
a_weight_db = [-70.4, -50.5, -30.2, -19.1, -10.9, -3.2, 0, 1.2, 1.0, -1.1, -6.6]

# Simple RMS calculation for relative dB
rms = np.sqrt(np.mean(audio**2))
db_relative = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)

print(f"Relative dB level: {db_relative:.2f} dB")

# For LAeq over time windows (e.g., 1-second windows)
window_size = sr  # 1 second
num_windows = len(audio) // window_size

for i in range(num_windows):
    window = audio[i*window_size:(i+1)*window_size]
    rms_window = np.sqrt(np.mean(window**2))
    db_window = 20 * np.log10(rms_window + 1e-10)
    print(f"Window {i+1}: {db_window:.2f} dB")
