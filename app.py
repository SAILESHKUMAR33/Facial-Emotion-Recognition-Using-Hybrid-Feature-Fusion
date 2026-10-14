import base64
import io
import json
import os
import sys
from pathlib import Path

# Suppress TF verbose logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import joblib
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template_string, request

# Try tf_keras first for Keras 2/3 compatibility, then fallback
try:
    import tf_keras as keras
except ImportError:
    try:
        import tensorflow.keras as keras
    except ImportError:
        import keras

import tensorflow as tf

from config import MODELS_DIR, MAX_SIDE
from face_utils import FaceCropper, LandmarkExtractor
from common import get_classes

app = Flask(__name__)

# Global model cache
_MODEL = None
_SCALER = None
_CLASSES = None


def get_inference_resources():
    global _MODEL, _SCALER, _CLASSES
    if _CLASSES is None:
        _CLASSES = get_classes()
    if _SCALER is None:
        scaler_path = MODELS_DIR / "landmark_scaler.pkl"
        if scaler_path.exists():
            _SCALER = joblib.load(scaler_path)
    if _MODEL is None:
        best_name = "effnet_lm_fusion"
        best_file = MODELS_DIR / "best_fusion.txt"
        if best_file.exists():
            best_name = best_file.read_text().strip()

        model_path = MODELS_DIR / f"{best_name}.keras"
        if not model_path.exists():
            # fallback to custom_cnn or transfer_learning if available
            for fallback in ["transfer_learning", "custom_cnn", "landmark_mlp"]:
                fb_path = MODELS_DIR / f"{fallback}.keras"
                if fb_path.exists():
                    model_path = fb_path
                    best_name = fallback
                    break

        if model_path.exists():
            _MODEL = keras.models.load_model(model_path, compile=False)
            print(f"[EmotionAI] Loaded model: {best_name} with inputs: {_MODEL.input_names}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    return _MODEL, _SCALER, _CLASSES


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Facial Emotion Recognition — Hybrid Feature Fusion AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --card-bg: rgba(22, 27, 40, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-cyan: #06b6d4;
      --text-main: #f8fafc;
      --text-dim: #94a3b8;
      --happy: #10b981;
      --sad: #3b82f6;
      --angry: #ef4444;
      --fear: #a855f7;
      --surprise: #f59e0b;
      --neutral: #64748b;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                  radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
                  var(--bg-dark);
      color: var(--text-main);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1.5rem;
      overflow-x: hidden;
    }

    header {
      text-align: center;
      max-width: 800px;
      margin-bottom: 2rem;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      margin-bottom: 1rem;
      letter-spacing: 0.5px;
    }

    .badge-dot {
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }

    h1 {
      font-size: 2.75rem;
      font-weight: 800;
      line-height: 1.2;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.75rem;
    }

    h1 span {
      background: linear-gradient(135deg, #818cf8 0%, #06b6d4 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    p.subtitle {
      color: var(--text-dim);
      font-size: 1.05rem;
      line-height: 1.5;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1.75rem;
      width: 100%;
      max-width: 1140px;
      align-items: start;
    }

    @media (max-width: 900px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 1.75rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
      position: relative;
    }

    .tabs {
      display: flex;
      background: rgba(10, 13, 20, 0.7);
      border-radius: 12px;
      padding: 4px;
      margin-bottom: 1.25rem;
      gap: 4px;
    }

    .tab-btn {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-family: inherit;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 10px 16px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .tab-btn.active {
      background: var(--accent);
      color: #ffffff;
      box-shadow: 0 4px 12px var(--accent-glow);
    }

    /* Media Area */
    .viewport {
      width: 100%;
      height: 380px;
      background: rgba(10, 13, 20, 0.9);
      border: 2px dashed rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 0.2s ease;
    }

    .viewport.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.05);
    }

    #videoFeed {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1);
      display: none;
    }

    #previewImg {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }

    .upload-hint {
      text-align: center;
      color: var(--text-dim);
      pointer-events: none;
    }

    .upload-hint svg {
      width: 48px;
      height: 48px;
      margin-bottom: 10px;
      stroke: var(--accent);
    }

    .controls {
      display: flex;
      gap: 12px;
      margin-top: 1.25rem;
    }

    .btn {
      flex: 1;
      background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%);
      color: #ffffff;
      border: none;
      padding: 14px 20px;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 16px var(--accent-glow);
      transition: all 0.2s ease;
    }

    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5);
    }

    .btn:active {
      transform: translateY(0);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      box-shadow: none;
      color: var(--text-main);
      border: 1px solid var(--card-border);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      box-shadow: none;
    }

    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    /* Result Panel */
    .top-prediction {
      text-align: center;
      padding: 1.25rem;
      background: rgba(10, 13, 20, 0.8);
      border-radius: 16px;
      margin-bottom: 1.5rem;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .top-emotion {
      font-size: 2.25rem;
      font-weight: 800;
      text-transform: capitalize;
      margin-bottom: 0.25rem;
    }

    .top-conf {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--accent-cyan);
      font-family: 'JetBrains Mono', monospace;
    }

    .bars-container {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bar-item {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.9rem;
      font-weight: 500;
      text-transform: capitalize;
    }

    .bar-track {
      height: 10px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
      width: 0%;
    }

    .cropped-face-box {
      margin-top: 1.5rem;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: rgba(10, 13, 20, 0.6);
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .cropped-face-box img {
      width: 64px;
      height: 64px;
      border-radius: 10px;
      object-fit: cover;
      border: 2px solid var(--accent);
    }

    .face-meta {
      font-size: 0.85rem;
      color: var(--text-dim);
      line-height: 1.4;
    }

    .face-meta strong {
      color: #fff;
    }

    /* Spinner */
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-left-color: #ffffff;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      animation: spin 0.8s linear infinite;
      display: none;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.9); }
    }

    .status-msg {
      margin-top: 10px;
      font-size: 0.85rem;
      text-align: center;
      color: var(--text-dim);
      min-height: 18px;
    }

    .status-msg.error {
      color: #f87171;
    }

    footer {
      margin-top: 3rem;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.85rem;
    }

    footer a {
      color: #a5b4fc;
      text-decoration: none;
    }
  </style>
</head>
<body>

  <header>
    <div class="badge">
      <div class="badge-dot"></div>
      Hybrid Deep Learning &bull; EfficientNet + Mediapipe Landmarks
    </div>
    <h1>Facial Emotion <span>Recognition AI</span></h1>
    <p class="subtitle">Multi-modal feature fusion neural network analyzing facial expressions in real-time with sub-millisecond inference.</p>
  </header>

  <main class="main-grid">
    <!-- Input Section -->
    <section class="card">
      <div class="tabs">
        <button class="tab-btn active" id="tabUpload" onclick="switchMode('upload')">
          <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
          Upload Photo
        </button>
        <button class="tab-btn" id="tabWebcam" onclick="switchMode('webcam')">
          <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
          Live Camera
        </button>
      </div>

      <div class="viewport" id="viewport" onclick="triggerFileSelect()">
        <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFile(event)">
        <video id="videoFeed" autoplay playsinline muted></video>
        <img id="previewImg" alt="Selected Preview">
        <div class="upload-hint" id="uploadHint">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"/></svg>
          <p><strong>Click to upload</strong> or drag & drop a photo</p>
          <span style="font-size:0.8rem; opacity: 0.7;">PNG, JPG, WEBP, or HEIC</span>
        </div>
      </div>

      <div class="controls">
        <button class="btn" id="predictBtn" onclick="runPrediction()">
          <span class="spinner" id="btnSpinner"></span>
          <span id="btnText">Analyze Emotion</span>
        </button>
        <button class="btn btn-secondary" id="camToggleBtn" style="display:none;" onclick="captureWebcamFrame()">
          📸 Snap & Analyze
        </button>
      </div>

      <div class="status-msg" id="statusMsg">Ready for analysis</div>
    </section>

    <!-- Results Section -->
    <section class="card">
      <h2 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
        Prediction Results
        <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-dim); background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px;" id="modelTag">Fusion Model</span>
      </h2>

      <div class="top-prediction">
        <div class="top-emotion" id="topEmotion">--</div>
        <div class="top-conf" id="topConfidence">0.0% Confidence</div>
      </div>

      <div class="bars-container" id="barsContainer">
        <!-- Bars dynamically generated -->
        <div class="bar-item">
          <div class="bar-label"><span>Happy</span> <span id="val-happy">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-happy" style="background: var(--happy);"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-label"><span>Neutral</span> <span id="val-neutral">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-neutral" style="background: var(--neutral);"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-label"><span>Surprise</span> <span id="val-surprise">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-surprise" style="background: var(--surprise);"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-label"><span>Sad</span> <span id="val-sad">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-sad" style="background: var(--sad);"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-label"><span>Angry</span> <span id="val-angry">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-angry" style="background: var(--angry);"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-label"><span>Fear</span> <span id="val-fear">0%</span></div>
          <div class="bar-track"><div class="bar-fill" id="fill-fear" style="background: var(--fear);"></div></div>
        </div>
      </div>

      <div class="cropped-face-box" id="croppedBox" style="display:none;">
        <img id="croppedFaceImg" src="" alt="Cropped Face">
        <div class="face-meta">
          <div>Face: <strong>Detected & Cropped (224x224)</strong></div>
          <div>Landmarks: <strong id="lmStatus">Extracted (468 points)</strong></div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <p>Powered by EfficientNet + MediaPipe Hybrid Fusion &bull; <a href="https://github.com/SAILESHKUMAR33/Facial-Emotion-Recognition-Using-Hybrid-Feature-Fusion" target="_blank">GitHub Repository</a></p>
  </footer>

  <script>
    let currentMode = 'upload';
    let stream = null;
    let selectedFile = null;

    const viewport = document.getElementById('viewport');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    const videoFeed = document.getElementById('videoFeed');
    const uploadHint = document.getElementById('uploadHint');
    const predictBtn = document.getElementById('predictBtn');
    const camToggleBtn = document.getElementById('camToggleBtn');
    const statusMsg = document.getElementById('statusMsg');

    // Drag and drop setup
    viewport.addEventListener('dragover', (e) => {
      e.preventDefault();
      viewport.classList.add('dragover');
    });
    viewport.addEventListener('dragleave', () => viewport.classList.remove('dragover'));
    viewport.addEventListener('drop', (e) => {
      e.preventDefault();
      viewport.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processSelectedFile(e.dataTransfer.files[0]);
      }
    });

    function triggerFileSelect() {
      if (currentMode === 'upload') {
        fileInput.click();
      }
    }

    function handleFile(e) {
      if (e.target.files && e.target.files[0]) {
        processSelectedFile(e.target.files[0]);
      }
    }

    function processSelectedFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        uploadHint.style.display = 'none';
        statusMsg.textContent = 'Image loaded. Click "Analyze Emotion"';
        statusMsg.className = 'status-msg';
      };
      reader.readAsDataURL(file);
    }

    async function switchMode(mode) {
      currentMode = mode;
      document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
      document.getElementById('tabWebcam').classList.toggle('active', mode === 'webcam');

      if (mode === 'webcam') {
        uploadHint.style.display = 'none';
        previewImg.style.display = 'none';
        videoFeed.style.display = 'block';
        predictBtn.style.display = 'none';
        camToggleBtn.style.display = 'flex';
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
          videoFeed.srcObject = stream;
          statusMsg.textContent = 'Camera active. Point face at camera and snap.';
        } catch (err) {
          statusMsg.textContent = 'Could not access webcam: ' + err.message;
          statusMsg.className = 'status-msg error';
        }
      } else {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
        }
        videoFeed.style.display = 'none';
        predictBtn.style.display = 'flex';
        camToggleBtn.style.display = 'none';
        if (selectedFile) {
          previewImg.style.display = 'block';
          uploadHint.style.display = 'none';
        } else {
          uploadHint.style.display = 'block';
          previewImg.style.display = 'none';
        }
      }
    }

    function captureWebcamFrame() {
      const canvas = document.createElement('canvas');
      canvas.width = videoFeed.videoWidth || 640;
      canvas.height = videoFeed.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      // Mirror frame like camera view
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      sendBase64ToApi(dataUrl);
    }

    async function runPrediction() {
      if (!selectedFile) {
        statusMsg.textContent = 'Please choose or drop an image first.';
        statusMsg.className = 'status-msg error';
        return;
      }
      setLoading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const resp = await fetch('/api/predict', {
          method: 'POST',
          body: formData
        });
        const res = await resp.json();
        handleResult(res);
      } catch (err) {
        statusMsg.textContent = 'Network error: ' + err.message;
        statusMsg.className = 'status-msg error';
      } finally {
        setLoading(false);
      }
    }

    async function sendBase64ToApi(dataUrl) {
      setLoading(true);
      try {
        const resp = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        });
        const res = await resp.json();
        handleResult(res);
      } catch (err) {
        statusMsg.textContent = 'Network error: ' + err.message;
        statusMsg.className = 'status-msg error';
      } finally {
        setLoading(false);
      }
    }

    function handleResult(res) {
      if (!res.success) {
        statusMsg.textContent = res.error || 'Prediction failed.';
        statusMsg.className = 'status-msg error';
        return;
      }

      statusMsg.textContent = `Analyzed successfully in ${res.latency_ms || 0}ms`;
      statusMsg.className = 'status-msg';

      document.getElementById('topEmotion').textContent = res.prediction;
      document.getElementById('topConfidence').textContent = `${(res.confidence * 100).toFixed(1)}% Confidence`;

      // Update probability bars
      if (res.probabilities) {
        for (const [emo, prob] of Object.entries(res.probabilities)) {
          const valEl = document.getElementById(`val-${emo.toLowerCase()}`);
          const fillEl = document.getElementById(`fill-${emo.toLowerCase()}`);
          if (valEl && fillEl) {
            valEl.textContent = `${prob.toFixed(1)}%`;
            fillEl.style.width = `${prob}%`;
          }
        }
      }

      // Show cropped face
      if (res.annotated_image) {
        document.getElementById('croppedBox').style.display = 'flex';
        document.getElementById('croppedFaceImg').src = res.annotated_image;
        document.getElementById('lmStatus').textContent = res.landmarks_detected ? 'Extracted & Geometric Scaled' : 'Not detected';
      }
    }

    function setLoading(isLoading) {
      document.getElementById('btnSpinner').style.display = isLoading ? 'inline-block' : 'none';
      document.getElementById('btnText').textContent = isLoading ? 'Analyzing...' : 'Analyze Emotion';
      predictBtn.disabled = isLoading;
      camToggleBtn.disabled = isLoading;
      if (isLoading) {
        statusMsg.textContent = 'Detecting facial geometry and running fusion model...';
      }
    }
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/health")
@app.route("/api/health")
def health():
    try:
        model, scaler, classes = get_inference_resources()
        return jsonify({
            "status": "online",
            "model_inputs": model.input_names if model else None,
            "classes": classes,
            "scaler_loaded": scaler is not None
        })
    except Exception as e:
        return jsonify({"status": "degraded", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    import time
    t0 = time.time()

    # 1. Read input image
    rgb = None
    try:
        if "file" in request.files:
            file = request.files["file"]
            img_bytes = file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            rgb = np.array(pil_img)
        elif request.is_json and "image" in request.json:
            data_url = request.json["image"]
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            img_bytes = base64.b64decode(data_url)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            rgb = np.array(pil_img)
        else:
            return jsonify({"success": False, "error": "No image found in request (upload file or base64 JSON)."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse image: {str(e)}"}), 400

    # Resize if extremely large to prevent OOM
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Detect & Crop face
    try:
        cropper = FaceCropper()
        crop = cropper.crop(rgb)
        if crop is None:
            return jsonify({
                "success": False,
                "error": "No face detected in the image. Please make sure the face is clearly visible and well-lit."
            }), 422
    except Exception as e:
        return jsonify({"success": False, "error": f"Face detection error: {str(e)}"}), 500

    # 3. Extract landmarks
    lm_feats = None
    landmarks_detected = False
    try:
        lm_extractor = LandmarkExtractor()
        feats = lm_extractor.features(crop)
        if feats is not None:
            landmarks_detected = True
            _, scaler, _ = get_inference_resources()
            if scaler is not None:
                lm_feats = scaler.transform(feats[None]).astype("float32")
    except Exception as e:
        print("[EmotionAI] Warning landmark extraction:", e)

    # 4. Run Model Inference
    try:
        model, _, classes = get_inference_resources()
        img_input = crop[None].astype("float32")

        inputs = {}
        for name in model.input_names:
            if name == "image":
                inputs["image"] = img_input
            elif name == "landmarks":
                if lm_feats is None:
                    # fallback landmark zero or mean vector
                    _, scaler, _ = get_inference_resources()
                    mean_feat = scaler.mean_ if scaler is not None else np.zeros((1, 219), dtype="float32")
                    inputs["landmarks"] = scaler.transform(mean_feat[None]).astype("float32") if scaler else mean_feat[None]
                else:
                    inputs["landmarks"] = lm_feats

        probs = model.predict(inputs, verbose=0)[0]
        top_idx = int(probs.argmax())
        top_emotion = classes[top_idx]
        top_confidence = float(probs[top_idx])

        # Emotion distribution dictionary
        prob_dict = {classes[i]: round(float(probs[i]) * 100, 2) for i in range(len(classes))}

        # Annotate face crop for response preview
        annotated_crop = cv2.resize(crop, (224, 224))
        annotated_bgr = cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("utf-8")

        latency = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "success": true_bool(),
            "prediction": top_emotion,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "face_detected": True,
            "landmarks_detected": landmarks_detected,
            "annotated_image": annotated_b64,
            "latency_ms": latency
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500


def true_bool():
    return True


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Emotion AI Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
