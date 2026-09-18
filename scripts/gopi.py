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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAMES_PER_VIDEO = 4
MAX_AUDIO_LEN = 256
MODEL_PATH = "deepfake_model_swin_cnn_acc.pth"

FFMPEG_BIN = r"C:\Users\abmeg\Downloads\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"


# -------------------------
# MODEL
# -------------------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # MUST match training: pretrained=False
        self.cnn = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
        self.cnn.head = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.cnn.forward_features(dummy)
            self.visual_feat_dim = feat.mean([-2, -1]).shape[1]

        # Audio CNN (same as training)
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
        raise ValueError("Cannot read video")

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

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.stack(frames)


# -------------------------
# AUDIO EXACT MEL-SPECTROGRAM (as in training)
# -------------------------
def extract_audio_mel(path):
    temp = "temp.wav"

    cmd = [FFMPEG_BIN, "-y", "-i", path, "-ar", "16000", "-ac", "1", temp]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    y, sr = librosa.load(temp, sr=16000)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel_tensor = torch.tensor(mel).float().unsqueeze(0)  # (1, 64, T)

    # Pad/truncate to MAX_AUDIO_LEN
    if mel_tensor.shape[2] < MAX_AUDIO_LEN:
        pad = MAX_AUDIO_LEN - mel_tensor.shape[2]
        mel_tensor = nn.functional.pad(mel_tensor, (0, pad))
    else:
        mel_tensor = mel_tensor[:, :, :MAX_AUDIO_LEN]

    mel_tensor = mel_tensor.unsqueeze(0)  # (1,1,64,256)
    mel_tensor = mel_tensor.to(DEVICE)

    os.remove(temp)
    return mel_tensor


# -------------------------
# PREDICT
# -------------------------
def predict_video(video_path):
    print("\n🎥 Processing:", video_path)

    frames = extract_frames(video_path).unsqueeze(0).to(DEVICE)
    audio = extract_audio_mel(video_path)  # already shaped correctly

    model = DeepfakeModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        out = model(frames, audio)
        prob = torch.softmax(out, dim=1)[0]
        pred = torch.argmax(prob).item()

    label = "REAL" if pred == 0 else "FAKE"
    confidence = prob[pred].item()

    print("\n🧾 RESULT:", label)
    print("📌 Confidence:", round(confidence, 4))

    return label, confidence


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    path = r"E:\PolyGlotFake\gopi.mp4"
    predict_video(path)
