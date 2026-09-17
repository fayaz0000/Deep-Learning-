# BirdCLEF+ 2026 — Code Demonstration
# LAB / DEMONSTRATION ONLY
# This code demonstrates the workflow and produces report-ready images.
# It does NOT claim an official Kaggle score, rank, or personal submission.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
METADATA = DATA_DIR / "train_metadata.csv"
AUDIO_DIR = DATA_DIR / "train_audio"

# ------------------------------------------------------------
# 1. LOAD METADATA
# ------------------------------------------------------------
if METADATA.exists():
    df = pd.read_csv(METADATA)
    print("Metadata shape:", df.shape)
    print(df.head())
else:
    # Demo data: replace this with the real Kaggle metadata.
    df = pd.DataFrame({
        "primary_label": [
            "species_a", "species_a", "species_b",
            "species_c", "species_b", "species_a"
        ],
        "filename": ["a1.ogg", "a2.ogg", "b1.ogg",
                     "c1.ogg", "b2.ogg", "a3.ogg"]
    })
    print("Using demonstration metadata.")

# ------------------------------------------------------------
# 2. IMAGE: SPECIES DISTRIBUTION
# ------------------------------------------------------------
counts = df["primary_label"].value_counts()

plt.figure(figsize=(10, 5))
counts.head(20).sort_values().plot(kind="barh")
plt.title("BirdCLEF+ 2026 — Species Distribution")
plt.xlabel("Number of audio clips")
plt.ylabel("Species")
plt.tight_layout()
plt.savefig("01_species_distribution.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# 3. LOAD A 5-SECOND AUDIO WINDOW
# ------------------------------------------------------------
# Real-data example:
#
# import librosa
# audio_file = next(AUDIO_DIR.rglob("*.ogg"))
# y, sr = librosa.load(audio_file, sr=32000, mono=True)

# Synthetic signal keeps this demonstration runnable without audio files.
sr = 32000
duration = 5
t = np.linspace(0, duration, sr * duration, endpoint=False)
rng = np.random.default_rng(7)
y = (0.25 * np.sin(2*np.pi*2500*t) +
     0.08 * rng.normal(size=len(t)))

# ------------------------------------------------------------
# 4. IMAGE: 5-SECOND WAVEFORM
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(y))/sr, y)
plt.title("5-Second Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig("02_waveform.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# 5. IMAGE: LOG-MEL SPECTROGRAM
# ------------------------------------------------------------
try:
    import librosa
    import librosa.display

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=16000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(
        mel_db, sr=sr, x_axis="time",
        y_axis="mel", fmax=16000
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("03_log_mel_spectrogram.png", dpi=200)
    plt.show()

except ImportError:
    print("Install librosa: pip install librosa")

# ------------------------------------------------------------
# 6. PREPROCESSING: 5-SECOND WINDOW
# ------------------------------------------------------------
WINDOW_SECONDS = 5
WINDOW_SAMPLES = sr * WINDOW_SECONDS

clip = y[:WINDOW_SAMPLES]
if len(clip) < WINDOW_SAMPLES:
    clip = np.pad(clip, (0, WINDOW_SAMPLES-len(clip)))

print("Sample rate:", sr)
print("Window:", WINDOW_SECONDS, "seconds")
print("Samples:", len(clip))

# ------------------------------------------------------------
# 7. IMAGE: PREPROCESSING PIPELINE
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")

steps = [
    "Continuous audio", "5-sec window", "Normalize",
    "Log-mel spectrogram", "CNN classifier", "Probabilities"
]

for i, step in enumerate(steps):
    ax.text(
        i/(len(steps)-1), 0.5, step,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6",
                  facecolor="white", edgecolor="black")
    )

plt.title("BirdCLEF+ 2026 — Audio Preprocessing Workflow")
plt.tight_layout()
plt.savefig("04_preprocessing_pipeline.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# 8. LIGHTWEIGHT CNN ARCHITECTURE
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class BirdCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x.flatten(1))

    model = BirdCNN(num_classes=10)
    print(model)

except ImportError:
    print("PyTorch is optional: pip install torch")

# ------------------------------------------------------------
# 9. IMAGE: EXAMPLE SPECIES PROBABILITIES
# ------------------------------------------------------------
species = ["Species A", "Species B", "Species C", "Species D", "Species E"]
probabilities = np.array([0.71, 0.12, 0.08, 0.05, 0.04])

plt.figure(figsize=(9, 4))
plt.barh(species[::-1], probabilities[::-1])
plt.xlim(0, 1)
plt.xlabel("Predicted probability")
plt.title("Example 5-Second Species Prediction")
plt.tight_layout()
plt.savefig("05_prediction_probabilities.png", dpi=200)
plt.show()

print("Demo images saved:")
for f in [
    "01_species_distribution.png",
    "02_waveform.png",
    "03_log_mel_spectrogram.png",
    "04_preprocessing_pipeline.png",
    "05_prediction_probabilities.png"
]:
    print(" -", f)

print("\nThe prediction values are illustrative only.")
