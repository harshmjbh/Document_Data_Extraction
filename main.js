// main.js
// JavaScript glue code for the document scanner WebAssembly module and
// minimal browser UI (camera preview, overlay, capture, and upload).

// The Emscripten-generated loader should be available as createScannerModule
// from scanner.js. Make sure scanner.js is included before this script.

let ScannerModulePromise = null;
let ScannerModule = null;

async function initScannerModule() {
  if (ScannerModule) return ScannerModule;
  if (!ScannerModulePromise) {
    // createScannerModule is defined by scanner.js (Emscripten output).
    ScannerModulePromise = createScannerModule();
  }
  ScannerModule = await ScannerModulePromise;
  return ScannerModule;
}

// Call the WASM detect_document function.
async function runDetect(imageData) {
  const Module = await initScannerModule();
  const width = imageData.width;
  const height = imageData.height;
  const numBytes = width * height * 4;

  const inPtr = Module._malloc(numBytes);
  Module.HEAPU8.set(imageData.data, inPtr);

  const cornersBytes = 8 * 4; // 8 floats
  const confBytes = 4;        // 1 float
  const cornersPtr = Module._malloc(cornersBytes);
  const confPtr = Module._malloc(confBytes);

  const found = Module._detect_document(
    inPtr,
    width,
    height,
    cornersPtr,
    confPtr
  );

  let result = null;
  if (found) {
    const cornersView = new Float32Array(
      Module.HEAPF32.buffer,
      cornersPtr,
      8
    );
    const confView = new Float32Array(
      Module.HEAPF32.buffer,
      confPtr,
      1
    );

    result = {
      corners: [
        { x: cornersView[0], y: cornersView[1] }, // TL
        { x: cornersView[2], y: cornersView[3] }, // TR
        { x: cornersView[4], y: cornersView[5] }, // BR
        { x: cornersView[6], y: cornersView[7] }, // BL
      ],
      confidence: confView[0],
    };
  }

  Module._free(inPtr);
  Module._free(cornersPtr);
  Module._free(confPtr);

  return result;
}

// Call the WASM warp_document function.
async function runWarp(imageData, corners, outWidth, outHeight) {
  const Module = await initScannerModule();
  const inBytes = imageData.width * imageData.height * 4;
  const outBytes = outWidth * outHeight * 4;

  const inPtr = Module._malloc(inBytes);
  Module.HEAPU8.set(imageData.data, inPtr);

  const cornersPtr = Module._malloc(8 * 4);
  const cornersView = new Float32Array(Module.HEAPF32.buffer, cornersPtr, 8);
  cornersView[0] = corners[0].x; cornersView[1] = corners[0].y;
  cornersView[2] = corners[1].x; cornersView[3] = corners[1].y;
  cornersView[4] = corners[2].x; cornersView[5] = corners[2].y;
  cornersView[6] = corners[3].x; cornersView[7] = corners[3].y;

  const outPtr = Module._malloc(outBytes);

  const ok = Module._warp_document(
    inPtr,
    imageData.width,
    imageData.height,
    cornersPtr,
    outPtr,
    outWidth,
    outHeight
  );

  let resultImageData = null;
  if (ok) {
    const outView = new Uint8ClampedArray(
      Module.HEAPU8.buffer,
      outPtr,
      outBytes
    );
    resultImageData = new ImageData(outView, outWidth, outHeight);
  }

  Module._free(inPtr);
  Module._free(cornersPtr);
  Module._free(outPtr);

  return resultImageData;
}

// Minimal UI wiring below.

const videoEl = document.getElementById("cameraVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas ? overlayCanvas.getContext("2d") : null;
const resultCanvas = document.getElementById("resultCanvas");
const resultCtx = resultCanvas ? resultCanvas.getContext("2d") : null;

const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const fileInput = document.getElementById("fileInput");

let detectCanvas = document.createElement("canvas");
let detectCtx = detectCanvas.getContext("2d");
let detectionRunning = false;
let latestDetection = null;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    videoEl.srcObject = stream;
    await videoEl.play();

    overlayCanvas.width = videoEl.videoWidth;
    overlayCanvas.height = videoEl.videoHeight;

    detectionRunning = true;
    window.requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error("Error starting camera:", err);
    alert("Unable to access camera. Check permissions.");
  }
}

async function detectionLoop() {
  if (!detectionRunning) return;
  if (videoEl.readyState < 2) {
    window.requestAnimationFrame(detectionLoop);
    return;
  }

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const maxSide = 640;
  const scale = Math.min(maxSide / vw, maxSide / vh, 1);

  detectCanvas.width = vw * scale;
  detectCanvas.height = vh * scale;

  detectCtx.drawImage(videoEl, 0, 0, detectCanvas.width, detectCanvas.height);
  const frame = detectCtx.getImageData(0, 0, detectCanvas.width, detectCanvas.height);

  try {
    const det = await runDetect(frame);
    latestDetection = det;

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (det && det.corners && det.corners.length === 4) {
      const sx = overlayCanvas.width / detectCanvas.width;
      const sy = overlayCanvas.height / detectCanvas.height;
      const color = det.confidence > 0.5 ? "lime" : "orange";

      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 3;
      overlayCtx.beginPath();
      overlayCtx.moveTo(det.corners[0].x * sx, det.corners[0].y * sy);
      for (let i = 1; i < 4; i++) {
        overlayCtx.lineTo(det.corners[i].x * sx, det.corners[i].y * sy);
      }
      overlayCtx.closePath();
      overlayCtx.stroke();
    }
  } catch (err) {
    console.error("Detection error:", err);
  }

  // Throttle to ~10 FPS
  setTimeout(() => window.requestAnimationFrame(detectionLoop), 100);
}

async function captureFromCamera() {
  if (!latestDetection || !latestDetection.corners || latestDetection.corners.length !== 4) {
    alert("No document detected to capture.");
    return;
  }

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = vw;
  captureCanvas.height = vh;
  const cctx = captureCanvas.getContext("2d");
  cctx.drawImage(videoEl, 0, 0, vw, vh);

  // Map detection corners from detectCanvas space to full-res.
  const scaleX = vw / detectCanvas.width;
  const scaleY = vh / detectCanvas.height;
  const fullCorners = latestDetection.corners.map(p => ({
    x: p.x * scaleX,
    y: p.y * scaleY,
  }));

  const frame = cctx.getImageData(0, 0, vw, vh);

  // Estimate output size from corner distances.
  const dist = (a, b) => {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const widthTop = dist(fullCorners[0], fullCorners[1]);
  const widthBottom = dist(fullCorners[3], fullCorners[2]);
  const maxWidth = Math.max(widthTop, widthBottom);

  const heightLeft = dist(fullCorners[0], fullCorners[3]);
  const heightRight = dist(fullCorners[1], fullCorners[2]);
  const maxHeight = Math.max(heightLeft, heightRight);

  let outWidth = Math.round(maxWidth);
  let outHeight = Math.round(maxHeight);

  const maxSide = 1500;
  const longest = Math.max(outWidth, outHeight);
  if (longest > maxSide) {
    const s = maxSide / longest;
    outWidth = Math.round(outWidth * s);
    outHeight = Math.round(outHeight * s);
  }

  const warped = await runWarp(frame, fullCorners, outWidth, outHeight);
  if (!warped) {
    alert("Warp failed.");
    return;
  }

  resultCanvas.width = warped.width;
  resultCanvas.height = warped.height;
  resultCtx.putImageData(warped, 0, 0);
}

async function handleFileUpload(ev) {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = async () => {
    const maxSide = 1600;
    const scale = Math.min(maxSide / img.width, maxSide / img.height, 1);

    const canvas = document.createElement("canvas");
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const det = await runDetect(frame);
    if (!det || !det.corners || det.corners.length !== 4) {
      alert("No document detected in uploaded image.");
      return;
    }

    // Estimate output size as above.
    const dist = (a, b) => {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return Math.sqrt(dx * dx + dy * dy);
    };

    const widthTop = dist(det.corners[0], det.corners[1]);
    const widthBottom = dist(det.corners[3], det.corners[2]);
    const maxWidth = Math.max(widthTop, widthBottom);

    const heightLeft = dist(det.corners[0], det.corners[3]);
    const heightRight = dist(det.corners[1], det.corners[2]);
    const maxHeight = Math.max(heightLeft, heightRight);

    let outWidth = Math.round(maxWidth);
    let outHeight = Math.round(maxHeight);

    const maxWarpSide = 1500;
    const longest = Math.max(outWidth, outHeight);
    if (longest > maxWarpSide) {
      const s = maxWarpSide / longest;
      outWidth = Math.round(outWidth * s);
      outHeight = Math.round(outHeight * s);
    }

    const warped = await runWarp(frame, det.corners, outWidth, outHeight);
    if (!warped) {
      alert("Warp failed for uploaded image.");
      return;
    }

    resultCanvas.width = warped.width;
    resultCanvas.height = warped.height;
    resultCtx.putImageData(warped, 0, 0);
  };

  img.src = URL.createObjectURL(file);
}

if (startCameraBtn) {
  startCameraBtn.addEventListener("click", startCamera);
}
if (captureBtn) {
  captureBtn.addEventListener("click", captureFromCamera);
}
if (fileInput) {
  fileInput.addEventListener("change", handleFileUpload);
}

