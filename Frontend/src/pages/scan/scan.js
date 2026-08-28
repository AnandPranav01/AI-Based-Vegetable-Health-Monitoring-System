import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';
import { analyzeVegetable, saveResult } from '/src/js/mock-api.js';
import { initTabs, showToast } from '/src/js/utils.js';

requireAuth();
injectSidebar('scan');
initTabs('.tabs');

// ────── CAMERA SECTION ──────
let stream = null;
let facingMode = 'environment';
const video = document.getElementById('cameraStream');
const canvas = document.getElementById('captureCanvas');
const previewImg = document.getElementById('previewImage');
const previewEmpty = document.getElementById('previewEmpty');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const startBtn = document.getElementById('startCameraBtn');
const stopBtn = document.getElementById('stopCameraBtn');
const switchBtn = document.getElementById('switchCameraBtn');
const cameraError = document.getElementById('cameraError');
let capturedName = '';
let capturedBlob = null;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio: false });
    video.srcObject = stream;
    startBtn.style.display = 'none';
    stopBtn.style.display = '';
    switchBtn.style.display = '';
    cameraError.style.display = 'none';
  } catch (err) {
    cameraError.style.display = 'block';
    cameraError.textContent = '⚠️ Camera access denied or unavailable. Please allow camera permission and try again.';
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  startBtn.style.display = '';
  stopBtn.style.display = 'none';
  switchBtn.style.display = 'none';
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
switchBtn.addEventListener('click', () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  stopCamera(); startCamera();
});

captureBtn.addEventListener('click', () => {
  if (!stream) { showToast('Start the camera first', 'warning'); return; }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg');
  previewImg.src = dataUrl;
  previewImg.style.display = 'block';
  previewEmpty.style.display = 'none';
  analyzeBtn.disabled = false;
  capturedName = 'camera_capture';
  // Convert canvas to blob for real API upload
  canvas.toBlob(blob => { capturedBlob = blob; }, 'image/jpeg', 0.92);
  showToast('Image captured! 📸');
});

analyzeBtn.addEventListener('click', () => runAnalysis(capturedBlob || capturedName, capturedName, analyzeBtn, 'analyzeBtnText', 'analyzeSpinner'));

// ────── UPLOAD SECTION ──────
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const uploadImg = document.getElementById('uploadPreviewImage');
const uploadEmpty = document.getElementById('uploadPreviewEmpty');
const uploadFileName = document.getElementById('uploadFileName');
const analyzeUploadBtn = document.getElementById('analyzeUploadBtn');
let uploadedName = '';
let uploadedFile = null;

uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
  else showToast('Please drop an image file', 'warning');
});
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  uploadedName = file.name;
  uploadedFile = file; // store for real API upload
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadImg.src = e.target.result;
    uploadImg.style.display = 'block';
    uploadEmpty.style.display = 'none';
    uploadFileName.textContent = `📄 ${file.name}`;
    uploadFileName.style.display = 'block';
    analyzeUploadBtn.disabled = false;
  };
  reader.readAsDataURL(file);
  showToast('Image loaded! Ready to analyze 🌿');
}

// Sample chips
document.getElementById('sampleChips').querySelectorAll('.chip[data-veg]').forEach(chip => {
  chip.addEventListener('click', () => {
    const veg = chip.dataset.veg;
    const emoji = chip.textContent.trim().split(' ')[0];
    uploadImg.src = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='400' height='300'><rect width='400' height='300' fill='%23141f17'/><text x='50%25' y='50%25' font-size='120' dominant-baseline='middle' text-anchor='middle'>${emoji}</text><text x='50%25' y='75%25' fill='%238ba898' font-size='24' dominant-baseline='middle' text-anchor='middle'>${veg.toUpperCase()}</text></svg>`;
    uploadImg.style.display = 'block';
    uploadEmpty.style.display = 'none';
    uploadFileName.textContent = `🧪 Sample: ${chip.textContent.trim()}`;
    uploadFileName.style.display = 'block';
    uploadedName = veg;
    uploadedFile = null; // samples use mock fallback
    analyzeUploadBtn.disabled = false;
    showToast(`Sample ${chip.textContent.trim()} loaded`);
  });
});

analyzeUploadBtn.addEventListener('click', () => runAnalysis(uploadedFile || uploadedName, uploadedName, analyzeUploadBtn, 'analyzeUploadBtnText', 'analyzeUploadSpinner'));

// ────── SHARED ANALYZE ──────
async function runAnalysis(imageSource, name, btn, textId, spinnerId) {
  btn.disabled = true;
  document.getElementById(textId).textContent = '🔬 Analyzing…';
  document.getElementById(spinnerId).style.display = 'inline-flex';

  try {
    const result = await analyzeVegetable(imageSource);
    saveResult(result);
    document.getElementById(textId).textContent = '✅ Done! Redirecting…';
    showToast(`Analysis complete: ${result.name} — ${result.status} 🎉`);
    setTimeout(() => window.location.href = '/src/pages/results/index.html', 900);
  } catch (err) {
    document.getElementById(textId).textContent = '🔬 Analyze';
    document.getElementById(spinnerId).style.display = 'none';
    btn.disabled = false;
    showToast(err.message || 'Analysis failed. Please try again.', 'error');
  }
}
