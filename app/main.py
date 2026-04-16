from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
import io
import base64
from .model import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Taka Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Use actual model prediction
        detections = predict_image(image_np)

        return JSONResponse({"detections": detections})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webcam")
async def webcam():
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Taka Detection - Webcam</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Arial, sans-serif;
                background: #1a1a1a;
                color: #fff;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                max-width: 900px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
            }
            h1 { text-align: center; margin-bottom: 20px; color: #4CAF50; }
            .content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .video-section {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            video {
                width: 100%;
                border-radius: 8px;
                background: #000;
                border: 2px solid #4CAF50;
            }
            canvas { display: none; }
            button {
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: background 0.3s;
            }
            button:hover { background: #45a049; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .results {
                background: #1a1a1a;
                padding: 15px;
                border-radius: 8px;
                max-height: 600px;
                overflow-y: auto;
                border: 2px solid #4CAF50;
            }
            .detection {
                background: #333;
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
            }
            .detection strong { color: #4CAF50; }
            .confidence { color: #ffeb3b; }
            #status {
                padding: 10px;
                text-align: center;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            .status-active { background: #4CAF50; color: white; }
            .status-inactive { background: #f44336; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Taka Detection - Real-time Webcam</h1>
            <div class="content">
                <div class="video-section">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas"></canvas>
                    <button id="toggleBtn" onclick="togglePrediction()">Start Detection</button>
                    <div id="status" class="status-inactive">Detection: OFF</div>
                </div>
                <div class="results">
                    <h3 style="color: #4CAF50; margin-bottom: 10px;">📊 Detections</h3>
                    <div id="detections">No detections yet...</div>
                </div>
            </div>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const toggleBtn = document.getElementById('toggleBtn');
            const status = document.getElementById('status');
            const detectionsDiv = document.getElementById('detections');
            let isDetecting = false;

            // Get webcam access
            async function startWebcam() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'user' }
                    });
                    video.srcObject = stream;
                } catch (err) {
                    alert('Cannot access webcam: ' + err);
                }
            }

            // Toggle detection
            function togglePrediction() {
                isDetecting = !isDetecting;
                toggleBtn.textContent = isDetecting ? 'Stop Detection' : 'Start Detection';
                status.textContent = isDetecting ? 'Detection: ON ✓' : 'Detection: OFF';
                status.className = isDetecting ? 'status-active' : 'status-inactive';
                
                if (isDetecting) {
                    detectFrame();
                }
            }

            // Capture and send frame for prediction
            async function detectFrame() {
                if (!isDetecting) return;

                const ctx = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);

                const imageData = canvas.toDataURL('image/jpeg');
                const base64 = imageData.split(',')[1];

                try {
                    const response = await fetch('/predict-base64', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64 })
                    });

                    const data = await response.json();
                    displayDetections(data.detections);
                } catch (err) {
                    console.error('Prediction error:', err);
                }

                // Process next frame after a short delay
                setTimeout(detectFrame, 500);
            }

            // Display detections
            function displayDetections(detections) {
                if (!detections || detections.length === 0) {
                    detectionsDiv.innerHTML = '<p style="color: #999;">No objects detected</p>';
                    return;
                }

                let html = '';
                detections.forEach((det, idx) => {
                    html += `
                        <div class="detection">
                            <div><strong>Object ${idx + 1}:</strong> ${det.class}</div>
                            <div class="confidence">Confidence: ${(det.confidence * 100).toFixed(2)}%</div>
                            <div style="font-size: 12px; color: #aaa;">
                                Bbox: [${det.bbox.map(b => b.toFixed(2)).join(', ')}]
                            </div>
                        </div>
                    `;
                });
                detectionsDiv.innerHTML = html;
            }

            // Initialize webcam on load
            window.onload = startWebcam;
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.post("/predict-base64")
async def predict_base64(data: dict):
    """Accept base64-encoded image and return predictions"""
    try:
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="No image data")

        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # Get predictions
        detections = predict_image(image_np)

        return JSONResponse({"detections": detections})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
