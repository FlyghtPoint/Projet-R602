import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.min.js';

// Charger les modèles de face-api.js
async function loadModels() {
    await faceapi.loadTinyFaceDetectorModel('/static/models');
    await faceapi.loadFaceExpressionModel('/static/models');
    await faceapi.loadFaceLandmarkModel('/static/models');
}

// Démarrer la webcam et la détection
async function startVideo() {
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve();
        };
    });
}

// Détecter les expressions faciales
async function detectExpressions() {
    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const context = overlay.getContext('2d');

    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(overlay, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        context.clearRect(0, 0, overlay.width, overlay.height);
        faceapi.draw.drawDetections(overlay, resizedDetections);
        faceapi.draw.drawFaceLandmarks(overlay, resizedDetections);
        faceapi.draw.drawFaceExpressions(overlay, resizedDetections);

        updateEmotionDisplay(resizedDetections);
    }, 100);
}

function updateEmotionDisplay(detections) {
    if (detections.length > 0) {
        const emotions = detections[0].expressions;
        const dominantEmotion = Object.keys(emotions).reduce((a, b) => emotions[a] > emotions[b] ? a : b);

        // Mettre à jour les barres de progression des émotions
        Object.keys(emotions).forEach(emotion => {
            const progressBar = document.getElementById(`${emotion}-bar`);
            if (progressBar) {
                const percentage = Math.round(emotions[emotion] * 100);
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
                document.getElementById(`${emotion}-value`).textContent = `${percentage}%`;
            }
        });

        // Mettre à jour l'émotion dominante
        document.getElementById('dominant-emotion').textContent = dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);
    }
}

// Initialiser l'application
async function init() {
    await loadModels();
    await startVideo();
    detectExpressions();
}

init();
