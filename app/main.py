from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
from models.pipeline import DefectDetectionPipeline

app = FastAPI(
    title="Real-Time Defect Detection API",
    description="Two-stage CV pipeline using YOLOv8 + EfficientNet for manufacturing QA",
    version="1.0.0"
)

# Initialize pipeline
pipeline = DefectDetectionPipeline()

class DetectionResponse(BaseModel):
    defect_type: str
    confidence: float
    defect_found: bool
    bbox: list
    message: str

@app.get("/")
def root():
    return {
        "message": "Defect Detection API",
        "status": "running",
        "model": "YOLOv8 + EfficientNet-B4"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile = File(...)):
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Run detection pipeline
        result = pipeline.run(img)

        return DetectionResponse(
            defect_type=result["defect_type"],
            confidence=result["confidence"],
            defect_found=result["defect_found"],
            bbox=result["bbox"],
            message=f"Defect detected: {result['defect_type']}" 
                    if result["defect_found"] 
                    else "No defect detected"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
def get_classes():
    return {
        "defect_classes": [
            "scratch",
            "crack", 
            "dent",
            "contamination",
            "corrosion",
            "pass"
        ]
    }
