import os
import cv2
import numpy as np
import soundfile as sf
import moviepy.editor as mpy
from tqdm import tqdm
import librosa
import gc

# ---------------- SETTINGS ----------------
DATASET_DIR = r"E:\PolyGlotFake\PolyGlotFake"  # Root dataset
FRAME_OUTPUT_DIR = os.path.join(DATASET_DIR, "frames")
AUDIO_OUTPUT_DIR = os.path.join(DATASET_DIR, "audio")
FRAMES_PER_VIDEO = 16      # Number of frames to extract per video
AUDIO_SR = 16000           # Sampling rate for audio

# ---------------- FUNCTIONS ----------------
def extract_frames(video_path, output_dir, num_frames=16):
    """Extract evenly spaced frames from a video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"⚠️ Video has 0 frames: {video_path}")
        cap.release()
        return

    frame_indices = np.linspace(0, frame_count-1, num=num_frames, dtype=int)
    
    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(os.path.join(output_dir, f"frame_{saved:04d}.jpg"), frame)
        saved += 1
    cap.release()
    cv2.destroyAllWindows()
    gc.collect()

def extract_audio(video_path, output_file, sr=16000):
    """Extract audio from video safely using MoviePy + librosa."""
    try:
        clip = mpy.VideoFileClip(video_path)
        if clip.audio is None:
            print(f"⚠️ No audio in {video_path}")
            clip.close()
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        temp_wav = output_file.replace(".wav", "_temp.wav")
        
        # Step 1: Save temporary audio via MoviePy
        clip.audio.write_audiofile(temp_wav, fps=sr, verbose=False, logger=None)
        clip.close()

        # Step 2: Load with librosa (always returns 1D array)
        y, _ = librosa.load(temp_wav, sr=sr, mono=True)
        sf.write(output_file, y, sr)

        # Step 3: Delete temporary file
        os.remove(temp_wav)
        gc.collect()

    except Exception as e:
        print(f"❌ Audio extraction failed for {video_path}: {e}")

# ---------------- MAIN EXTRACTION ----------------
for cls in ["real", "fake"]:
    cls_dir = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"⚠️ Missing class folder: {cls_dir}")
        continue

    for lang in os.listdir(cls_dir):
        lang_path = os.path.join(cls_dir, lang)
        if not os.path.isdir(lang_path):
            continue
        print(f"\nProcessing {cls}/{lang}")
        videos = [v for v in os.listdir(lang_path) if v.lower().endswith((".mp4", ".avi", ".mov"))]
        for vid in tqdm(videos, desc=f"{cls}/{lang}"):
            video_path = os.path.join(lang_path, vid)
            base_name = os.path.splitext(vid)[0]
            
            # Frames output
            frame_dir = os.path.join(FRAME_OUTPUT_DIR, cls, lang, base_name)
            extract_frames(video_path, frame_dir, num_frames=FRAMES_PER_VIDEO)
            
            # Audio output
            audio_file = os.path.join(AUDIO_OUTPUT_DIR, cls, lang, f"{base_name}.wav")
            extract_audio(video_path, audio_file, sr=AUDIO_SR)

print("\n✅ Extraction completed successfully!")
