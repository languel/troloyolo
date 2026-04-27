import { AutoModel, AutoProcessor, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// DOM Elements
const video = document.getElementById('video');
const imageSource = document.getElementById('image-source');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const videoContainer = document.getElementById('video-container');
const startBtn = document.getElementById('start-btn');
const btnIcon = document.getElementById('btn-icon');
const btnText = document.getElementById('btn-text');
const modelSelect = document.getElementById('model-select');
const backendSelect = document.getElementById('backend-select');
const sourceControl = document.getElementById('source-control');
const sourceSelect = document.getElementById('source-select');
const mediaFile = document.getElementById('media-file');
const mediaUrl = document.getElementById('media-url');
const mediaUrlRow = document.getElementById('media-url-row');
const loadUrlBtn = document.getElementById('load-url-btn');
const layerCamera = document.getElementById('layer-camera');
const layerDetect = document.getElementById('layer-detect');
const layerSegment = document.getElementById('layer-segment');
const layerPose = document.getElementById('layer-pose');
const layerObb = document.getElementById('layer-obb');
const layerClassify = document.getElementById('layer-classify');
const toggleFacing = document.getElementById('toggle-facing');
const toggleMirror = document.getElementById('toggle-mirror');
const thresholdInput = document.getElementById('threshold');
const thresholdValueEl = document.getElementById('threshold-value');
const fpsEl = document.getElementById('fps');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const uiToggle = document.getElementById('ui-toggle');

// State
const models = {};
let processor = null;
let isRunning = false;
let isProcessing = false;
let threshold = 0.65;
let layers = {
  camera: true,
  detect: true,
  segment: false,
  pose: true,
  obb: false,
  classify: false
};
let backend = 'auto';
let sourceMode = 'camera';
let activeSourceType = 'camera';
let mediaObjectUrl = null;
let lastFrameErrorMessage = '';
let facingMode = 'environment';
let mirrorVideo = false;
let activeDevice = null;
let animationId = null;
let isUiCollapsed = false;
const urlParams = new URLSearchParams(window.location.search);

// Offscreen canvas for frame capture
const offscreen = document.createElement('canvas');
const offscreenCtx = offscreen.getContext('2d');
const yoloInput = document.createElement('canvas');
const yoloInputCtx = yoloInput.getContext('2d', { willReadFrequently: true });

// Constants
const COLORS = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6', '#ef4444', '#10b981', '#3b82f6'];
const CLASS_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16]
];
const POSE_THRESHOLD = 0.25;
const YOLO_INPUT_SIZE = 640;
const YOLO_MASK_ALPHA = 0.36;
const AXERA_SEGMENT_REPO = 'https://huggingface.co/AXERA-TECH/yolo26-seg/resolve/main';
const DEVICE_CONFIG = {
  webgpu: { dtypes: ['fp16', 'fp32'], label: 'WebGPU' },
  wasm: { dtypes: ['q8', 'fp32'], label: 'WASM' }
};
const MODEL_LAYERS = {
  detect: {
    label: 'detection',
    runtime: 'transformers',
    modelId: (baseId) => baseId,
    parse: parseDetectionOutput
  },
  segment: {
    label: 'segmentation',
    runtime: 'onnxruntime',
    modelId: (baseId) => `${AXERA_SEGMENT_REPO}/yolo26${getSelectedModelSize(baseId)}-seg_640x640.onnx`,
    parse: parseAxeraSegmentationOutput
  },
  pose: {
    label: 'pose',
    runtime: 'transformers',
    modelId: (baseId) => baseId.replace('-ONNX', '-pose-ONNX'),
    parse: parsePoseOutput
  },
  obb: {
    label: 'oriented boxes',
    runtime: 'transformers',
    modelId: (baseId) => baseId.replace('-ONNX', '-obb-ONNX'),
    parse: parseObbOutput
  },
  classify: {
    label: 'classification',
    runtime: 'transformers',
    modelId: (baseId) => baseId.replace('-ONNX', '-cls-ONNX'),
    parse: parseClassificationOutput
  }
};

// UI Helpers
const setStatus = (text, type = 'default') => {
  statusText.textContent = text;
  statusDot.className = 'status-dot ' + type;
};

const showLoader = (text) => {
  loaderText.textContent = text;
  loader.classList.add('visible');
};

const hideLoader = () => loader.classList.remove('visible');

const hasCameraSecurityContext = () => window.isSecureContext || ['localhost', '127.0.0.1', '::1'].includes(location.hostname);

const getPixelRatio = () => Math.max(1, window.devicePixelRatio || 1);

const getActiveMediaElement = () => activeSourceType === 'image' ? imageSource : video;

const getActiveMediaSize = () => activeSourceType === 'image'
  ? { width: imageSource.naturalWidth, height: imageSource.naturalHeight }
  : { width: video.videoWidth, height: video.videoHeight };

const isYoutubeUrl = (value) => {
  try {
    const { hostname } = new URL(value);
    return /(^|\.)youtube\.com$/i.test(hostname) || /(^|\.)youtu\.be$/i.test(hostname);
  } catch {
    return false;
  }
};

const isGifSource = (value = '') => /\.gif(?:[?#].*)?$/i.test(value);

const revokeMediaObjectUrl = () => {
  if (mediaObjectUrl) URL.revokeObjectURL(mediaObjectUrl);
  mediaObjectUrl = null;
};

const setElementCrossOrigin = (element, url) => {
  if (url.startsWith('blob:')) {
    element.removeAttribute('crossorigin');
    return;
  }

  element.crossOrigin = 'anonymous';
};

const setActiveMediaVisibility = () => {
  const imageActive = activeSourceType === 'image';
  video.classList.toggle('media-source', imageActive);
  video.classList.toggle('active', !imageActive);
  imageSource.classList.toggle('active', imageActive);
  video.classList.toggle('hidden-layer', !layers.camera && !imageActive);
  imageSource.classList.toggle('hidden-layer', !layers.camera && imageActive);
  video.classList.toggle('mirrored', mirrorVideo && !imageActive);
  imageSource.classList.toggle('mirrored', mirrorVideo && imageActive);
};

const getSourceLabel = () => ({
  camera: 'Camera',
  video: 'Video',
  image: 'GIF'
}[activeSourceType] || 'Source');

const getFrameReadErrorMessage = (error) => {
  if (error?.name === 'SecurityError') {
    return 'This media source blocks canvas reads. Try a local file or a direct URL with CORS enabled.';
  }

  return error?.message || 'Could not process this media frame.';
};

const resizeCanvas = () => {
  const rect = canvas.getBoundingClientRect();
  const width = Math.round(rect.width * getPixelRatio());
  const height = Math.round(rect.height * getPixelRatio());

  if (width > 0 && height > 0 && (canvas.width !== width || canvas.height !== height)) {
    canvas.width = width;
    canvas.height = height;
  }
};

const prepareFrameCanvas = () => {
  const { width, height } = getActiveMediaSize();
  if (!width || !height) return false;

  if (offscreen.width !== width || offscreen.height !== height) {
    offscreen.width = width;
    offscreen.height = height;
  }

  resizeCanvas();
  return true;
};

const getVideoCoverTransform = () => {
  const { width: mediaWidth, height: mediaHeight } = getActiveMediaSize();
  const sourceWidth = offscreen.width || mediaWidth || canvas.width;
  const sourceHeight = offscreen.height || mediaHeight || canvas.height;
  const scale = Math.max(canvas.width / sourceWidth, canvas.height / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;

  return {
    scale,
    offsetX: (canvas.width - width) / 2,
    offsetY: (canvas.height - height) / 2
  };
};

const mapPointToCanvas = (x, y, transform) => ({
  x: (mirrorVideo ? offscreen.width - x : x) * transform.scale + transform.offsetX,
  y: y * transform.scale + transform.offsetY
});

const mapBoxToCanvas = ([x, y, w, h], transform) => ({
  x: (mirrorVideo ? offscreen.width - x - w : x) * transform.scale + transform.offsetX,
  y: y * transform.scale + transform.offsetY,
  w: w * transform.scale,
  h: h * transform.scale
});

const setLayerAvailable = (key, available) => {
  const input = {
    detect: layerDetect,
    segment: layerSegment,
    pose: layerPose,
    obb: layerObb,
    classify: layerClassify
  }[key];

  if (!input) return;
  input.disabled = false;
  input.closest('.toggle')?.classList.toggle('unavailable', !available);
  input.closest('.toggle')?.setAttribute('title', available ? '' : 'Layer model is not available for this model/backend yet');
  if (!available) input.checked = false;
  layers[key] = available && input.checked;
};

const getEnabledModelLayerKeys = () => Object.keys(MODEL_LAYERS).filter(key => layers[key]);

const getModelLayerKeysToLoad = () => getEnabledModelLayerKeys();

const getSelectedModelSize = (modelId) => {
  const match = modelId.match(/yolo26([nslmx])/i);
  return match?.[1]?.toLowerCase() || 'n';
};

const getTransformersLayerKeys = () => getEnabledModelLayerKeys().filter(key => MODEL_LAYERS[key].runtime === 'transformers');

const hasRunnableLayer = () => getEnabledModelLayerKeys().some(key => models[key]);

const getOutputTensor = (output) => {
  const values = Object.values(output);
  return values.find(value => value?.data && value?.dims) || values.find(value => value?.data) || null;
};

const getLabel = (model, classId) => model?.config?.id2label?.[classId] || model?.model?.config?.id2label?.[classId] || CLASS_LABELS[classId] || `Class ${classId}`;

const sigmoid = (value) => 1 / (1 + Math.exp(-value));

const scaleModelCoordinate = (value, axisSize) => {
  if (Math.abs(value) <= 1.5) return value * axisSize;
  return value * axisSize / YOLO_INPUT_SIZE;
};

const scaleModelLength = (value, axisSize) => {
  if (Math.abs(value) <= 1.5) return value * axisSize;
  return value * axisSize / YOLO_INPUT_SIZE;
};

const scaleModelBox = ([x1, y1, x2, y2]) => {
  const left = scaleModelCoordinate(x1, offscreen.width);
  const top = scaleModelCoordinate(y1, offscreen.height);
  const right = scaleModelCoordinate(x2, offscreen.width);
  const bottom = scaleModelCoordinate(y2, offscreen.height);
  return [left, top, right - left, bottom - top];
};

const scaleModelCenterBox = ([cx, cy, w, h]) => [
  scaleModelCoordinate(cx - w / 2, offscreen.width),
  scaleModelCoordinate(cy - h / 2, offscreen.height),
  scaleModelLength(w, offscreen.width),
  scaleModelLength(h, offscreen.height)
];

const iou = (a, b) => {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(1, areaA + areaB - intersection);
};

const nms = (detections, iouThreshold = 0.7, maxDetections = 30) => {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const keep = [];

  for (const detection of sorted) {
    if (keep.length >= maxDetections) break;
    if (keep.every(selected => iou(detection.xyxy, selected.xyxy) < iouThreshold)) {
      keep.push(detection);
    }
  }

  return keep;
};

const hexToRgba = (hex, alpha) => {
  const value = hex.replace('#', '');
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const setUiCollapsed = (collapsed) => {
  isUiCollapsed = collapsed;
  document.documentElement.classList.toggle('ui-collapsed', collapsed);
  document.body.classList.toggle('ui-collapsed', collapsed);
  document.body.classList.remove('ui-quiet');
  uiToggle.setAttribute('aria-expanded', String(!collapsed));
  uiToggle.setAttribute('aria-label', collapsed ? 'Show controls' : 'Hide controls');
};

setUiCollapsed(urlParams.has('clean') || urlParams.get('ui') === '0');

const getDeviceCandidates = () => {
  if (backend === 'wasm') return ['wasm'];
  if (backend === 'webgpu') return ['webgpu', 'wasm'];
  return navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
};

const disposeModels = async () => {
  await Promise.all(Object.values(models).map(model => model?.dispose?.() || model?.session?.release?.()));
  for (const key of Object.keys(models)) delete models[key];
  processor = null;
};

async function loadSingleModel(modelId, device, dtypeCandidates, label, progressCallback) {
  let lastError = null;

  for (const dtype of dtypeCandidates) {
    try {
      return await AutoModel.from_pretrained(modelId, {
        device,
        dtype,
        progress_callback: progressCallback(`${label} (${DEVICE_CONFIG[device].label}, ${dtype})`)
      });
    } catch (error) {
      lastError = error;
      console.warn(`${label} failed on ${device}/${dtype}:`, error);
    }
  }

  throw lastError;
}

async function loadOrtModel(modelUrl, device) {
  const executionProviders = device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders });

  return {
    runtime: 'onnxruntime',
    session,
    inputName: session.inputNames[0],
    outputNames: session.outputNames,
    dispose: () => session.release?.()
  };
}

async function loadLayerModels(modelId, device, progressCallback) {
  const config = DEVICE_CONFIG[device];
  const layerKeys = getModelLayerKeysToLoad();

  for (const key of layerKeys) {
    const layer = MODEL_LAYERS[key];
    const layerModelId = layer.modelId(modelId);

    try {
      showLoader(`Loading ${layer.label} model (${config.label})...`);
      models[key] = layer.runtime === 'onnxruntime'
        ? await loadOrtModel(layerModelId, device)
        : await loadSingleModel(
            layerModelId,
            device,
            config.dtypes,
            `Loading ${layer.label} model`,
            progressCallback
          );
      setLayerAvailable(key, true);
    } catch (error) {
      if (key === 'detect') throw error;

      console.warn(`${layer.label} layer unavailable for ${layerModelId}:`, error);
      setLayerAvailable(key, false);
    }
  }

  if (getTransformersLayerKeys().length) {
    showLoader('Loading processor...');
    processor = await AutoProcessor.from_pretrained(modelId);
  }
  activeDevice = device;
}

// Model Loading
async function loadModels(modelId) {
  try {
    if (isRunning) stopCamera(true);
    while (isProcessing) await new Promise(r => setTimeout(r, 50));

    await disposeModels();
    activeDevice = null;
    startBtn.disabled = true;

    const progressCallback = (label) => (info) => {
      if (info.status === 'progress' && info.file.endsWith('.onnx')) {
        showLoader(`${label} (${Math.round((info.loaded / info.total) * 100)}%)`);
      }
    };

    setStatus('Loading...', 'loading');
    const devices = getDeviceCandidates();
    let lastError = null;

    for (const device of devices) {
      try {
        await loadLayerModels(modelId, device, progressCallback);
        break;
      } catch (error) {
        lastError = error;
        console.warn(`Model loading failed on ${device}:`, error);
        await disposeModels();

        const canFallback = device === 'webgpu' && devices.includes('wasm');
        if (!canFallback) throw error;
        showLoader('WebGPU unavailable, switching to WASM...');
      }
    }

    if (getEnabledModelLayerKeys().length && !hasRunnableLayer()) {
      throw lastError || new Error('No enabled model layer is available');
    }

    const activeLabel = DEVICE_CONFIG[activeDevice].label;
    setStatus(`Ready (${activeLabel})`, 'ready');
    hideLoader();
    startBtn.disabled = false;
    startSource();
  } catch (error) {
    console.error('Model loading failed:', error);
    setStatus('Error', 'error');
    showLoader('Failed: ' + error.message);
  }
}

// Source Control
const setRunningUi = (running) => {
  isRunning = running;
  btnIcon.innerHTML = running
    ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l15 8-15 8V4z"/></svg>';
  btnText.textContent = running ? `Stop ${getSourceLabel()}` : `Start ${sourceMode === 'camera' ? 'Camera' : 'Source'}`;
  startBtn.classList.toggle('running', running);
  document.body.classList.toggle('camera-running', running);
};

function startProcessingLoop() {
  prepareFrameCanvas();
  lastFrameErrorMessage = '';
  setRunningUi(true);
  hideLoader();
  setStatus('Running', 'running');
  loop();
}

async function startSource() {
  if (sourceMode === 'file') {
    const file = mediaFile.files?.[0];
    if (!file) {
      mediaFile.click();
      return;
    }
    await startFileSource(file);
    return;
  }

  if (sourceMode === 'url') {
    await startUrlSource(mediaUrl.value.trim());
    return;
  }

  await startCamera();
}

async function startCamera() {
  try {
    stopMediaElements();
    activeSourceType = 'camera';
    setActiveMediaVisibility();

    if (!hasCameraSecurityContext()) {
      throw new Error('Camera access requires HTTPS or localhost. Use Tailscale Serve and open the https://*.ts.net URL.');
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera access is not available in this browser context.');
    }

    showLoader('Accessing camera...');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      startProcessingLoop();
    };
  } catch (error) {
    console.error('Camera error:', error);
    setStatus('Camera Error', 'error');
    showLoader(error.message || 'Camera access denied');
  }
}

async function startFileSource(file) {
  stopMediaElements();
  const url = URL.createObjectURL(file);
  mediaObjectUrl = url;
  const isGif = file.type === 'image/gif' || isGifSource(file.name);
  await startMediaUrl(url, isGif ? 'image' : 'video', file.name);
}

async function startUrlSource(url) {
  if (!url) {
    throw new Error('Enter a direct media URL first.');
  }

  if (isYoutubeUrl(url)) {
    throw new Error('YouTube links cannot be used directly in this static browser demo because the frames are cross-origin and not available to canvas. Use a downloaded video file or a direct MP4/WebM/GIF URL.');
  }

  stopMediaElements();
  await startMediaUrl(url, isGifSource(url) ? 'image' : 'video', url);
}

async function startMediaUrl(url, type, label) {
  showLoader(`Loading ${type === 'image' ? 'GIF' : 'video'}...`);
  activeSourceType = type;
  setActiveMediaVisibility();

  if (type === 'image') {
    await new Promise((resolve, reject) => {
      imageSource.onload = resolve;
      imageSource.onerror = () => reject(new Error(`Could not load GIF source: ${label}`));
      setElementCrossOrigin(imageSource, url);
      imageSource.src = url;
    });
    startProcessingLoop();
    return;
  }

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error(`Could not load video source: ${label}`));
    video.srcObject = null;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    setElementCrossOrigin(video, url);
    video.src = url;
    video.play().catch(reject);
  });

  startProcessingLoop();
}

function stopMediaElements() {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  video.pause();
  video.removeAttribute('src');
  video.removeAttribute('crossorigin');
  video.load();
  imageSource.removeAttribute('src');
  imageSource.removeAttribute('crossorigin');
  revokeMediaObjectUrl();
}

function stopCamera(keepProcessingFlag = false) {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  stopMediaElements();

  if (!keepProcessingFlag) isProcessing = false;

  setRunningUi(false);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const statusLabel = activeDevice ? `Ready (${DEVICE_CONFIG[activeDevice].label})` : 'Ready';
  setStatus(statusLabel, 'ready');
  fpsEl.textContent = '0';
}

// Detection Loop
function loop() {
  if (!isRunning) return;

  if (hasRunnableLayer() && !isProcessing) {
    isProcessing = true;
    const startTime = performance.now();
    detect()
      .then(() => fpsEl.textContent = Math.round(1000 / (performance.now() - startTime)))
      .catch((error) => {
        const message = getFrameReadErrorMessage(error);
        console.error('Frame processing error:', error);
        fpsEl.textContent = '0';

        if (message !== lastFrameErrorMessage) {
          lastFrameErrorMessage = message;
          setStatus('Frame Error', 'error');
          showLoader(message);
        }
      })
      .finally(() => isProcessing = false);
  } else if (!hasRunnableLayer()) {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  if (isRunning) animationId = requestAnimationFrame(loop);
}

async function detect() {
  if (!prepareFrameCanvas()) return;

  offscreenCtx.drawImage(getActiveMediaElement(), 0, 0, offscreen.width, offscreen.height);
  const detections = [];
  const activeLayerKeys = getEnabledModelLayerKeys().filter(key => models[key]);
  const transformerLayerKeys = activeLayerKeys.filter(key => MODEL_LAYERS[key].runtime === 'transformers');
  const ortLayerKeys = activeLayerKeys.filter(key => MODEL_LAYERS[key].runtime === 'onnxruntime');
  const results = [];

  if (transformerLayerKeys.length && processor) {
    const image = RawImage.fromCanvas(offscreen);
    const inputs = await processor(image);
    const transformerResults = await Promise.all(transformerLayerKeys.map(async key => [key, await models[key](inputs)]));
    results.push(...transformerResults);
  }

  if (ortLayerKeys.length) {
    const input = prepareYoloInput();
    const ortResults = await Promise.all(ortLayerKeys.map(async key => {
      const model = models[key];
      const feeds = { [model.inputName]: new ort.Tensor('float32', input.data, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]) };
      return [key, await model.session.run(feeds), input];
    }));
    results.push(...ortResults);
  }

  for (const [key, output, input] of results) {
    detections.push(...MODEL_LAYERS[key].parse(output, models[key], input));
  }

  if (isRunning) draw(detections);
}

function prepareYoloInput() {
  yoloInput.width = YOLO_INPUT_SIZE;
  yoloInput.height = YOLO_INPUT_SIZE;

  const scale = Math.min(YOLO_INPUT_SIZE / offscreen.height, YOLO_INPUT_SIZE / offscreen.width);
  const width = Math.round(offscreen.width * scale);
  const height = Math.round(offscreen.height * scale);

  yoloInputCtx.fillStyle = 'rgb(114, 114, 114)';
  yoloInputCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  yoloInputCtx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, width, height);

  const pixels = yoloInputCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).data;
  const data = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  const planeSize = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

  for (let i = 0; i < planeSize; i++) {
    const pixel = i * 4;
    data[i] = pixels[pixel] / 255;
    data[planeSize + i] = pixels[pixel + 1] / 255;
    data[planeSize * 2 + i] = pixels[pixel + 2] / 255;
  }

  return { data, scale, width, height, originalWidth: offscreen.width, originalHeight: offscreen.height };
}

function parseDetectionOutput(output, model) {
  const candidates = [];
  const scores = output.logits?.sigmoid?.().data;
  const boxes = output.pred_boxes?.data;
  if (!scores || !boxes) return [];

  const numBoxes = Math.min(300, boxes.length / 4);
  const numClasses = scores.length / numBoxes;

  for (let i = 0; i < numBoxes; i++) {
    let maxScore = 0, maxClass = 0;
    for (let j = 0; j < numClasses; j++) {
      const score = scores[i * numClasses + j];
      if (score > maxScore) { maxScore = score; maxClass = j; }
    }
    if (maxScore >= threshold) {
      const [cx, cy, w, h] = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
      const box = scaleModelCenterBox([cx, cy, w, h]);
      candidates.push({
        type: 'object',
        box,
        xyxy: [box[0], box[1], box[0] + box[2], box[1] + box[3]],
        score: maxScore,
        classId: maxClass,
        label: getLabel(model, maxClass)
      });
    }
  }

  return nms(candidates, 0.45, 12);
}

function parsePoseOutput(output) {
  const detections = [];
  const tensor = getOutputTensor(output);
  if (!tensor) return detections;

  const data = tensor.data;
  const stride = 57;
  const numBoxes = Math.floor(data.length / stride);

  for (let i = 0; i < numBoxes; i++) {
    const offset = i * stride;
    const score = data[offset + 4];
    if (score >= threshold) {
      const keypoints = [];
      for (let k = 0; k < 17; k++) {
        const kIdx = offset + 6 + k * 3;
        keypoints.push({
          x: scaleModelCoordinate(data[kIdx], offscreen.width),
          y: scaleModelCoordinate(data[kIdx + 1], offscreen.height),
          c: data[kIdx + 2]
        });
      }
      detections.push({
        type: 'pose',
        box: scaleModelBox([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]),
        score,
        keypoints
      });
    }
  }

  return detections;
}

function parseAxeraSegmentationOutput(output, model, input) {
  const tensors = model.outputNames.map(name => output[name]).filter(Boolean);
  if (tensors.length < 10) return [];

  const confRaw = -Math.log(1 / threshold - 1);
  const candidates = [];
  const strides = [8, 16, 32];

  for (let scaleIndex = 0; scaleIndex < strides.length; scaleIndex++) {
    const boxTensor = tensors[scaleIndex * 3];
    const classTensor = tensors[scaleIndex * 3 + 1];
    const maskTensor = tensors[scaleIndex * 3 + 2];
    if (!boxTensor || !classTensor || !maskTensor) continue;

    const [, gridH, gridW] = boxTensor.dims;
    const classCount = classTensor.dims.at(-1);
    const maskCoeffCount = maskTensor.dims.at(-1);
    const stride = strides[scaleIndex];

    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const cell = y * gridW + x;
        const classOffset = cell * classCount;
        let classId = 0;
        let rawScore = -Infinity;

        for (let c = 0; c < classCount; c++) {
          const score = classTensor.data[classOffset + c];
          if (score > rawScore) {
            rawScore = score;
            classId = c;
          }
        }

        if (rawScore < confRaw) continue;

        const boxOffset = cell * 4;
        const anchorX = x + 0.5;
        const anchorY = y + 0.5;
        const x1 = (anchorX - boxTensor.data[boxOffset]) * stride;
        const y1 = (anchorY - boxTensor.data[boxOffset + 1]) * stride;
        const x2 = (anchorX + boxTensor.data[boxOffset + 2]) * stride;
        const y2 = (anchorY + boxTensor.data[boxOffset + 3]) * stride;
        const maskOffset = cell * maskCoeffCount;

        candidates.push({
          xyxy: [x1, y1, x2, y2],
          modelBox: [x1, y1, x2, y2],
          score: sigmoid(rawScore),
          classId,
          maskCoeffs: maskTensor.data.slice(maskOffset, maskOffset + maskCoeffCount)
        });
      }
    }
  }

  const proto = tensors.at(-1);
  const selected = nms(candidates, 0.7, 20);

  return selected.map(det => {
    const [x1, y1, x2, y2] = det.xyxy;
    const box = [
      Math.max(0, x1 / input.scale),
      Math.max(0, y1 / input.scale),
      Math.min(input.originalWidth, x2 / input.scale) - Math.max(0, x1 / input.scale),
      Math.min(input.originalHeight, y2 / input.scale) - Math.max(0, y1 / input.scale)
    ];

    return {
      type: 'segment',
      box,
      modelBox: det.modelBox,
      mask: proto ? buildSegmentationMask(proto, det.maskCoeffs, det.modelBox, input) : null,
      score: det.score,
      classId: det.classId,
      label: getLabel(model, det.classId)
    };
  });
}

function buildSegmentationMask(proto, coeffs, modelBox, input) {
  const [, maskCount, protoH, protoW] = proto.dims;
  if (!maskCount || !protoH || !protoW) return null;

  const mask = new Uint8Array(protoH * protoW);
  const widthRatio = protoW / YOLO_INPUT_SIZE;
  const heightRatio = protoH / YOLO_INPUT_SIZE;
  const crop = {
    x1: Math.max(0, Math.floor(modelBox[0] * widthRatio)),
    y1: Math.max(0, Math.floor(modelBox[1] * heightRatio)),
    x2: Math.min(protoW, Math.ceil(modelBox[2] * widthRatio)),
    y2: Math.min(protoH, Math.ceil(modelBox[3] * heightRatio))
  };
  const validW = input.width * widthRatio;
  const validH = input.height * heightRatio;

  for (let y = crop.y1; y < crop.y2; y++) {
    if (y > validH) continue;
    for (let x = crop.x1; x < crop.x2; x++) {
      if (x > validW) continue;
      let value = 0;
      const pixel = y * protoW + x;
      for (let m = 0; m < maskCount; m++) {
        value += coeffs[m] * proto.data[m * protoH * protoW + pixel];
      }
      mask[pixel] = sigmoid(value) > 0.5 ? 1 : 0;
    }
  }

  return {
    data: mask,
    width: protoW,
    height: protoH,
    scale: input.scale,
    validWidth: validW,
    validHeight: validH
  };
}

function parseObbOutput(output, model) {
  const tensor = getOutputTensor(output);
  if (!tensor) return [];

  const data = tensor.data;
  const stride = tensor.dims?.at?.(-1) || 0;
  if (stride < 7) return [];

  const detections = [];
  const rows = Math.min(Math.floor(data.length / stride), 300);
  for (let i = 0; i < rows; i++) {
    const offset = i * stride;
    const score = data[offset + 4] > 1 ? sigmoid(data[offset + 4]) : data[offset + 4];
    if (score < threshold) continue;

    const classId = Math.max(0, Math.round(data[offset + 5]));
    detections.push({
      type: 'obb',
      cx: scaleModelCoordinate(data[offset], offscreen.width),
      cy: scaleModelCoordinate(data[offset + 1], offscreen.height),
      w: scaleModelLength(data[offset + 2], offscreen.width),
      h: scaleModelLength(data[offset + 3], offscreen.height),
      angle: data[offset + 6],
      score,
      classId,
      label: getLabel(model, classId)
    });
  }

  return detections;
}

function parseClassificationOutput(output, model) {
  const tensor = output.logits || getOutputTensor(output);
  if (!tensor?.data) return [];

  let maxScore = -Infinity;
  let classId = 0;
  for (let i = 0; i < tensor.data.length; i++) {
    if (tensor.data[i] > maxScore) {
      maxScore = tensor.data[i];
      classId = i;
    }
  }

  const score = maxScore > 1 ? sigmoid(maxScore) : maxScore;
  if (score < threshold) return [];

  return [{
    type: 'classification',
    score,
    classId,
    label: getLabel(model, classId)
  }];
}

// Drawing
function draw(detections) {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const transform = getVideoCoverTransform();
  const pixelRatio = canvas.width / (canvas.getBoundingClientRect().width || canvas.width);

  for (const det of detections.filter(d => d.type === 'segment')) {
    const { x, y, w, h } = mapBoxToCanvas(det.box, transform);
    const color = COLORS[det.classId % COLORS.length];
    const label = `${det.label} ${Math.round(det.score * 100)}%`;
    const lineWidth = 2 * pixelRatio;
    const fontSize = 12 * pixelRatio;
    const labelHeight = 18 * pixelRatio;
    const labelPadding = 4 * pixelRatio;

    if (det.mask) {
      drawSegmentationMask(det.mask, color, transform);
    } else {
      ctx.fillStyle = color + '33';
      ctx.fillRect(x, y, w, h);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([6 * pixelRatio, 4 * pixelRatio]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    ctx.font = `bold ${fontSize}px system-ui`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y > labelHeight ? y - labelHeight : y, tw + labelPadding * 2, labelHeight);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + labelPadding, y > labelHeight ? y - 5 * pixelRatio : y + 13 * pixelRatio);
  }

  for (const det of detections.filter(d => d.type === 'object')) {
    const { x, y, w, h } = mapBoxToCanvas(det.box, transform);
    const color = COLORS[det.classId % COLORS.length];
    const label = `${det.label} ${Math.round(det.score * 100)}%`;
    const lineWidth = 2 * pixelRatio;
    const fontSize = 12 * pixelRatio;
    const labelHeight = 18 * pixelRatio;
    const labelPadding = 4 * pixelRatio;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, w, h);

    ctx.font = `bold ${fontSize}px system-ui`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y > labelHeight ? y - labelHeight : y, tw + labelPadding * 2, labelHeight);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + labelPadding, y > labelHeight ? y - 5 * pixelRatio : y + 13 * pixelRatio);
  }

  for (const det of detections.filter(d => d.type === 'pose')) {
    ctx.lineWidth = 3 * pixelRatio;
    ctx.strokeStyle = '#22d3ee';
    for (const [i, j] of SKELETON) {
      const a = det.keypoints[i], b = det.keypoints[j];
      if (a?.c >= POSE_THRESHOLD && b?.c >= POSE_THRESHOLD) {
        const start = mapPointToCanvas(a.x, a.y, transform);
        const end = mapPointToCanvas(b.x, b.y, transform);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
    }

    for (const kp of det.keypoints) {
      if (kp.c < POSE_THRESHOLD) continue;
      const point = mapPointToCanvas(kp.x, kp.y, transform);
      ctx.fillStyle = '#6366f1';
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5 * pixelRatio, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2 * pixelRatio;
      ctx.stroke();
    }
  }

  for (const det of detections.filter(d => d.type === 'obb')) {
    const center = mapPointToCanvas(det.cx, det.cy, transform);
    const color = COLORS[det.classId % COLORS.length];
    const width = det.w * transform.scale;
    const height = det.h * transform.scale;
    const angle = mirrorVideo ? -det.angle : det.angle;

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * pixelRatio;
    ctx.strokeRect(-width / 2, -height / 2, width, height);
    ctx.restore();

    ctx.font = `bold ${12 * pixelRatio}px system-ui`;
    ctx.fillStyle = color;
    ctx.fillText(`${det.label} ${Math.round(det.score * 100)}%`, center.x + 6 * pixelRatio, center.y - 6 * pixelRatio);
  }

  const classification = detections.find(d => d.type === 'classification');
  if (classification) {
    const label = `${classification.label} ${Math.round(classification.score * 100)}%`;
    ctx.font = `bold ${18 * pixelRatio}px system-ui`;
    const padding = 10 * pixelRatio;
    const width = ctx.measureText(label).width + padding * 2;
    const height = 34 * pixelRatio;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.68)';
    ctx.fillRect(canvas.width - width - 16 * pixelRatio, 16 * pixelRatio, width, height);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, canvas.width - width - 16 * pixelRatio + padding, 39 * pixelRatio);
  }
}

function drawSegmentationMask(mask, color, transform) {
  const sourceCell = YOLO_INPUT_SIZE / mask.width / mask.scale;
  const step = canvas.width > 900 ? 2 : 3;

  ctx.fillStyle = hexToRgba(color, YOLO_MASK_ALPHA);
  for (let y = 0; y < mask.height; y += step) {
    if (y > mask.validHeight) continue;
    for (let x = 0; x < mask.width; x += step) {
      if (x > mask.validWidth || !mask.data[y * mask.width + x]) continue;
      const sourceX = x * YOLO_INPUT_SIZE / mask.width / mask.scale;
      const sourceY = y * YOLO_INPUT_SIZE / mask.height / mask.scale;
      const rect = mapBoxToCanvas([sourceX, sourceY, sourceCell * step, sourceCell * step], transform);
      ctx.fillRect(rect.x, rect.y, Math.max(1, rect.w), Math.max(1, rect.h));
    }
  }
}

// Event Listeners
const reportSourceError = (error) => {
  console.error('Source error:', error);
  setRunningUi(false);
  setStatus('Source Error', 'error');
  showLoader(error.message || 'Could not load source');
};

startBtn.addEventListener('click', () => {
  if (isRunning) {
    stopCamera();
    return;
  }
  startSource().catch(reportSourceError);
});
uiToggle.addEventListener('click', () => setUiCollapsed(!isUiCollapsed));
videoContainer.addEventListener('click', () => {
  if (isUiCollapsed) {
    document.body.classList.add('ui-quiet');
  } else {
    setUiCollapsed(true);
  }
});
document.addEventListener('keydown', (event) => {
  if (event.altKey && (event.code === 'KeyU' || event.key.toLowerCase() === 'u')) {
    event.preventDefault();
    setUiCollapsed(!isUiCollapsed);
  }
});
window.addEventListener('resize', resizeCanvas);
thresholdInput.addEventListener('input', (e) => {
  threshold = e.target.value / 100;
  thresholdValueEl.textContent = `${e.target.value}%`;
});
layerCamera.addEventListener('change', (e) => {
  layers.camera = e.target.checked;
  setActiveMediaVisibility();
});

sourceSelect.addEventListener('change', (e) => {
  sourceMode = e.target.value;
  if (isRunning) stopCamera();
  mediaFile.hidden = sourceMode !== 'file';
  mediaUrlRow.hidden = sourceMode !== 'url';
  toggleFacing.closest('.toggle').hidden = sourceMode !== 'camera';
  sourceControl.title = sourceMode === 'url'
    ? 'Use a direct MP4/WebM/GIF URL. YouTube pages cannot be read as canvas frames.'
    : 'Use a camera stream, local movie/GIF, or direct MP4/WebM/GIF URL.';
  setRunningUi(false);
});

mediaFile.addEventListener('change', () => {
  if (sourceMode === 'file' && mediaFile.files?.[0] && !isRunning) {
    startSource().catch(reportSourceError);
  }
});

loadUrlBtn.addEventListener('click', () => {
  if (isRunning) stopCamera();
  startSource().catch(reportSourceError);
});

mediaUrl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    loadUrlBtn.click();
  }
});

const bindModelLayerToggle = (input, key) => {
  input.addEventListener('change', async (event) => {
    layers[key] = event.target.checked;
    await loadModels(modelSelect.value);
  });
};

bindModelLayerToggle(layerDetect, 'detect');
bindModelLayerToggle(layerSegment, 'segment');
bindModelLayerToggle(layerPose, 'pose');
bindModelLayerToggle(layerObb, 'obb');
bindModelLayerToggle(layerClassify, 'classify');

toggleFacing.addEventListener('change', async (e) => {
  facingMode = e.target.checked ? 'user' : 'environment';
  if (isRunning && sourceMode === 'camera') {
    stopCamera(true);
    await startCamera();
  }
});
toggleMirror.addEventListener('change', (e) => {
  mirrorVideo = e.target.checked;
  setActiveMediaVisibility();
});
modelSelect.addEventListener('change', (e) => loadModels(e.target.value));
backendSelect.addEventListener('change', (e) => {
  backend = e.target.value;
  loadModels(modelSelect.value);
});

// Initialize
setActiveMediaVisibility();
loadModels(modelSelect.value);
