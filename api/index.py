from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/sakets-workspace-0e6tp/workflows/custom-workflow"

app = FastAPI(title="Hero Vida Damage Assessor API")

# Allow your React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================
@app.get("/")
def health():
    return JSONResponse({"status": "healthy", "message": "Hero Backend Running 🚀"})

# ==========================================
# AGENT 1: OpenCV PREPROCESSING (Pure Python)
# ==========================================
def agent1_quality_gate(image_np):
    """Checks for blur and applies CLAHE enhancement."""
    import cv2
    
    # Convert to grayscale for variance calculation
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Check (Variance of Laplacian)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_score < 100.0 # Threshold (adjust if needed)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This removes glare/shadows and enhances dents
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return is_blurry, blur_score, enhanced_img

# ==========================================
# MAIN PIPELINE ENDPOINT
# ==========================================
@app.post("/api/analyze")
async def analyze_vehicle(file: UploadFile = File(...)):
    import cv2
    import numpy as np
    
    # 1. Read the uploaded image into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Run Agent 1 (Quality Gate)
    is_blurry, blur_score, enhanced_img = agent1_quality_gate(img)
    
    if is_blurry:
        return {
            "status": "Fail", 
            "reason": f"Image Quality Gate Failed. Blur score ({blur_score:.2f}) is too low. Please retake photo."
        }

    # 3. Convert Enhanced Image back to Base64 for Roboflow
    _, buffer = cv2.imencode('.jpg', enhanced_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    # 4. Call Agents 2 & 3 (Roboflow Workflow)
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": base64_img}
        }
    }
    
    try:
        print("Agent 1 Pass. Sending Enhanced Image to Roboflow...")
        response = requests.post(ROBOFLOW_WORKFLOW_URL, json=payload)
        response.raise_for_status() # Check for HTTP errors
        roboflow_data = response.json()
        
        return {
            "status": "Pass",
            "blur_score": round(blur_score, 2),
            "roboflow_data": roboflow_data
        }
    except requests.exceptions.RequestException as e:
        print(f"Roboflow Error: {e}")
        if response.text:
            print(f"Details: {response.text}")
        raise HTTPException(status_code=500, detail="Error communicating with Roboflow.")
