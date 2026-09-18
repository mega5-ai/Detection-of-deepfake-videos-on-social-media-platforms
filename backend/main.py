from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import uuid
from backend.inference import predict_video

app = FastAPI(title="Deepfake Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_extension = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Inference
        # Pass the original filename explicitly because we renamed the file on disk to a UUID
        result = predict_video(file_path, original_filename=file.filename)
        
        # Cleanup
        os.remove(file_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
