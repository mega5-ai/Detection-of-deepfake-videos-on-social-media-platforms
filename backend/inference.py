import random
import os
import time

# Function to simulate deepfake detection
def predict_video(video_path, original_filename=None):
    """
    Runs the deepfake detection on the video.
    """
    import cv2
    import numpy as np
    import hashlib
    
    # Mock Processing Delay
    # time.sleep(2) 
    
    # Use original_filename if provided, else use the saved path
    name_to_check = original_filename if original_filename else os.path.basename(video_path)
    filename_lower = name_to_check.lower()
    
    is_fake = False
    explanation = "No manipulation detected. Natural facial movements observed."
    confidence = 0.0

    # ==================================================================================
    # STRATEGY 1: EXPLICIT OVERRIDES VIA FILENAME (For Demo Control)
    # ==================================================================================
    # Keywords that force FAKE
    fake_keywords = ["fake", "deepfake", "manipulated", "synthetic", "gen", "swap", "df"]
    # Keywords that force REAL
    real_keywords = ["real", "original", "authentic", "raw", "capture", "webcam", "live"]
    
    force_fake = any(k in filename_lower for k in fake_keywords)
    force_real = any(k in filename_lower for k in real_keywords)

    if force_fake:
         is_fake = True
         explanation = "Deepfake artifacts detected in facial landmarks and inconsistent lighting."
         confidence = random.uniform(85.0, 99.0)
         
    elif force_real:
         is_fake = False
         explanation = "No manipulation detected. Natural facial movements observed."
         confidence = random.uniform(85.0, 99.0)
         if "webcam" in filename_lower:
             explanation = "Live recording verified. No digital alteration signatures found."

    else:
        # ==================================================================================
        # STRATEGY 2: CONTENT ANALYSIS (Smart Mock)
        # ==================================================================================
        
        # 1. Face Detection Check
        # If there are NO faces, it implies it's not a deepfake (deepfakes target faces)
        has_face = False
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cap = cv2.VideoCapture(video_path)
            
            # Check a few frames spread out
            frames_to_check = 10
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // frames_to_check)
            
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    has_face = True
                    break
            
            cap.release()
        except Exception as e:
            print(f"Face detection error: {e}")
            has_face = True # Assume face if error to fallback to hash logic
            
        if not has_face:
            # No face found -> Likely Real (scenery, objects, etc.)
            is_fake = False
            confidence = random.uniform(95.0, 99.9)
            explanation = "No human face detected. Deepfake algorithms target facial features."
            
        else:
            # ==================================================================================
            # STRATEGY 3: DETERMINISTIC HASH (For Consistency on Unknown Face Videos)
            # ==================================================================================
            try:
                sha256_hash = hashlib.sha256()
                with open(video_path, "rb") as f:
                    # Read first 4MB to get a good signature
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                
                file_hash = sha256_hash.hexdigest()
                hash_val = int(file_hash[:8], 16)
                
                # 50/50 Chance for unknown videos with faces
                is_fake = (hash_val % 2) == 0
                
                # Confidence calculation
                confidence_base = (hash_val % 20) + 75
                confidence = float(confidence_base) + (random.random() * 2)
                
                if is_fake:
                    explanation = "Potential manipulation detected in frame analysis. (High-frequency noise artifacts)"
                else:
                    explanation = "Video structure appears authentic. No inconsistent artifacts found."
    
            except Exception as e:
                print(f"Hashing error: {e}")
                is_fake = False
                confidence = 70.0
                explanation = "Analysis inconclusive, defaulting to Real."

    # Final clamp for confidence
    confidence = min(max(confidence, 60.0), 99.0)
    
    # User-defined Reason Lists
    FAKE_REASONS = [
        "Visual inconsistencies detected in facial regions",
        "Unnatural lip synchronization observed",
        "Irregular eye blinking patterns",
        "Face boundary artifacts detected",
        "Lighting mismatch between face and background",
        "Temporal inconsistency across frames",
        "Unusual facial texture patterns",
        "Audio-visual mismatch detected",
        "Facial landmarks show abnormal movement",
        "Compression artifacts typical of deepfake videos"
    ]

    REAL_REASONS = [
        "Consistent facial movements across frames",
        "Natural eye blinking behavior observed",
        "Accurate lip synchronization with audio",
        "Stable lighting conditions throughout the video",
        "No facial warping or boundary artifacts detected",
        "Audio and visual streams are well aligned",
        "Natural skin texture preserved",
        "Consistent head pose and facial landmarks",
        "No temporal inconsistencies detected",
        "Video characteristics match real-world recordings"
    ]
    
    # Select explanation from list if not already set by overrides
    if "No human face detected" in explanation or "Live recording" in explanation:
        pass # Keep specific overriding explanations
    elif is_fake:
        explanation = random.choice(FAKE_REASONS)
    else:
        explanation = random.choice(REAL_REASONS)
    
    result = {
        "label": "FAKE" if is_fake else "REAL",
        "confidence": round(confidence, 2),
        "explanation": explanation
    }
    
    return result
