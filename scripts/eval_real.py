import torch
import torch.nn as nn
import numpy as np
import librosa
import cv2
import subprocess
import timm
from torchvision import transforms
from PIL import Image
import os

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAMES_PER_VIDEO = 4
MAX_AUDIO_LEN = 256
MODEL_PATH = "deepfake_model_swin_cnn_acc.pth"

# Set your FFmpeg path (VERY IMPORTANT)
FFMPEG_BIN = r"C:\Users\abmeg\Downloads\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"


# -------------------------
# MODEL
# -------------------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.cnn.head = nn.Identity()

        # Determine feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.cnn.forward_features(dummy)
            self.visual_feat_dim = feat.mean([-2, -1]).shape[1]

        # Audio CNN
        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.visual_feat_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, frames, audio):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)

        visual_feats = self.cnn.forward_features(frames)
        visual_feats = visual_feats.mean([-2, -1])
        visual_embeds = visual_feats.view(B, T, -1).mean(dim=1)

        audio_embeds = self.audio_net(audio)
        return self.fc(torch.concat([visual_embeds, audio_embeds], dim=1))


# -------------------------
# FRAME EXTRACTION
# -------------------------
def extract_frames(path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        raise ValueError("Cannot read video or empty video!")

    ids = np.linspace(0, total - 1, num_frames).astype(int)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    frames = []

    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        frames.append(transform(pil))

    cap.release()

    if len(frames) == 0:
        raise ValueError("Frame extraction failed!")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.stack(frames)


# -------------------------
# AUDIO EXTRACTION (FFmpeg)
# -------------------------
def extract_audio_mel(path):
    temp = "temp_audio.wav"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", path,
        "-ar", "16000",
        "-ac", "1",
        temp
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    audio, sr = librosa.load(temp, sr=16000)

    # Use 128 MFCCs (safe max)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
    audio_feat = mfcc.mean(axis=1)   # shape = (128,)

    # PAD TO 256
    if audio_feat.shape[0] < 256:
        pad = 256 - audio_feat.shape[0]
        audio_feat = np.pad(audio_feat, (0, pad))

    os.remove(temp)

    audio_feat = torch.tensor(audio_feat).float().view(1, 1, 16, 16)

    return audio_feat


# -------------------------
# PREDICTION
# -------------------------
def predict_video(video_path):
    print("\n🎥 Processing:", video_path)

    frames = extract_frames(video_path)
    audio = extract_audio_mel(video_path)

    frames = frames.unsqueeze(0).to(DEVICE)
    audio = audio.to(DEVICE)

    model = DeepfakeModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        out = model(frames, audio)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob).item()

    label = "REAL" if pred < 0.5 else "FAKE"
    confidence = float(prob[0][pred])

    print("\n🧾 RESULT:", label)
    print("📌 Confidence:", round(confidence, 4))

    return label, confidence


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    path = r"E:\PolyGlotFake\PolyGlotFake\real\es\es_1.mp4"
    predict_video(path)
