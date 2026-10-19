/* Skin Cancer AI frontend — talks to the FastAPI backend. */
const API = "http://localhost:5000";
const RECENT_LIMIT = 8;
const BUILTIN_SAMPLES = [
  { file: "sample-nevus.jpg", label: "Sample nevus" },
  { file: "sample-akiec.jpg", label: "Sample actinic keratosis" }
];

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropzoneHint = document.getElementById("dropzoneHint");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const errorMsg = document.getElementById("errorMsg");
const resultsCard = document.getElementById("resultsCard");
const resultsHeading = document.querySelector(".results-head h2");
const predictionLabel = document.getElementById("predictionLabel");
const meaningLine = document.getElementById("meaningLine");
const cautionNote = document.getElementById("cautionNote");
const confidenceBar = document.querySelector(".confidence-bar");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const confidenceBadge = document.getElementById("confidenceBadge");
const heatmapBox = document.querySelector(".heatmap-box");
const classList = document.getElementById("classList");
const heatmapImg = document.getElementById("heatmapImg");
const heatmapLoading = document.getElementById("heatmapLoading");
const backendStatus = document.getElementById("backendStatus");
const backendStatusText = document.getElementById("backendStatusText");
const recentSection = document.getElementById("recentSection");
const recentList = document.getElementById("recentList");

const classMeanings = {
  "actinic keratosis": "A rough patch caused by sun damage. Not cancer itself, but it can slowly turn into skin cancer — worth showing a doctor.",
  "basal cell carcinoma": "The most common type of skin cancer. It grows slowly and rarely spreads, but it does need treatment — see a doctor.",
  "benign keratosis": "A harmless, wart-like growth that often appears with age. Not cancer.",
  "dermatofibroma": "A small, firm, harmless bump in the skin. Not cancer.",
  melanoma: "The most serious type of skin cancer. It can spread if not caught early — show this to a doctor promptly.",
  nevus: "A common mole. Usually harmless — see a doctor if it changes in size, shape or colour.",
  "vascular lesion": "A harmless growth of tiny blood vessels, like a cherry-red spot. Not cancer."
};

let currentFile = null;
let previewUrl = null;
let recentResults = [];
let activeResultId = null;

/* ---------- backend health ---------- */
async function checkBackend() {
  try {
    const res = await fetch(`${API}/diagnose`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    backendStatus.classList.add("online");
    backendStatus.classList.remove("offline");
    backendStatusText.textContent = `backend online · ${data.class_count} classes`;
  } catch {
    backendStatus.classList.add("offline");
    backendStatus.classList.remove("online");
    backendStatusText.textContent = "backend offline — start it on :5000";
  }
}
checkBackend();
setInterval(checkBackend, 15000);

/* ---------- upload ---------- */
dropzone.addEventListener("click", (event) => {
  if (event.target !== fileInput) fileInput.click();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
["dragover", "dragenter"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files[0]) {
    currentFile = event.dataTransfer.files[0];
    showPreview(currentFile);
  }
});
fileInput.addEventListener("change", () => {
  currentFile = fileInput.files[0] || null;
  if (currentFile) showPreview(currentFile);
});

function showPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.classList.remove("hidden");
  dropzoneHint.classList.add("hidden");
  analyzeBtn.disabled = false;
  clearBtn.disabled = false;
  hideError();
  resultsCard.classList.add("hidden");
  activeResultId = null;
  renderRecentResults();
}

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  currentFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  preview.classList.add("hidden");
  dropzoneHint.classList.remove("hidden");
  resultsCard.classList.add("hidden");
  analyzeBtn.disabled = true;
  clearBtn.disabled = true;
  activeResultId = null;
  renderRecentResults();
  hideError();
});

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.classList.remove("hidden");
}
function hideError() {
  errorMsg.textContent = "";
  errorMsg.classList.add("hidden");
}

/* ---------- sample images ---------- */
const sampleRow = document.querySelector(".sample-row");
const sampleStatus = sampleRow.querySelector(".sample-status");
const sampleButtons = BUILTIN_SAMPLES.map(({ file, label }) => {
  const button = document.createElement("button");
  button.className = "btn sample-btn";
  button.type = "button";
  button.dataset.sample = file;
  button.textContent = label;
  sampleRow.insertBefore(button, sampleStatus);
  return button;
});

sampleButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    sampleStatus.textContent = "";
    button.disabled = true;
    try {
      const response = await fetch(`samples/${button.dataset.sample}`);
      if (!response.ok) {
        sampleStatus.textContent = "sample not added yet";
        return;
      }
      const blob = await response.blob();
      const file = new File([blob], button.dataset.sample, { type: blob.type || "image/jpeg" });
      currentFile = file;
      fileInput.value = "";
      showPreview(file);
      await analyzeFile(file);
    } catch {
      sampleStatus.textContent = "sample not added yet";
    } finally {
      button.disabled = false;
    }
  });
});

/* ---------- analysis ---------- */
analyzeBtn.addEventListener("click", () => {
  if (currentFile) analyzeFile(currentFile);
});

async function analyzeFile(file) {
  hideError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  heatmapLoading.classList.remove("hidden");
  heatmapImg.classList.add("hidden");

  try {
    const form = new FormData();
    form.append("image", file);

    const response = await fetch(`${API}/predict-all`, { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || body.error || `HTTP ${response.status}`);
    }
    const data = await response.json();
    let heatmap = null;

    if (Number(data.confidence) >= 0.3) {
      // Heatmap failure should not hide an otherwise successful prediction.
      try {
        const heatmapForm = new FormData();
        heatmapForm.append("image", file);
        const heatmapResponse = await fetch(`${API}/explain`, { method: "POST", body: heatmapForm });
        if (heatmapResponse.ok) {
          const heatmapData = await heatmapResponse.json();
          heatmap = heatmapData.heatmap;
        } else {
          console.warn(`Heatmap request failed: HTTP ${heatmapResponse.status}`);
        }
      } catch (error) {
        console.warn("Heatmap request failed", error);
      }
    }

    if (Number(data.confidence) < 0.3) {
      activeResultId = null;
      renderResults(data, null);
      renderRecentResults();
      return;
    }

    const id = `${Date.now()}-${Math.random()}`;
    const record = {
      id,
      file,
      imageUrl: URL.createObjectURL(file),
      data,
      heatmap,
      createdAt: new Date()
    };
    recentResults.unshift(record);
    if (recentResults.length > RECENT_LIMIT) {
      const removed = recentResults.pop();
      URL.revokeObjectURL(removed.imageUrl);
    }
    renderSavedResult(record);
    renderRecentResults();
  } catch (error) {
    showError("Could not analyze the image: " + error.message +
      ". Make sure the backend is running (uvicorn main:app --port 5000).");
  } finally {
    heatmapLoading.classList.add("hidden");
    analyzeBtn.disabled = !currentFile;
    analyzeBtn.textContent = "Analyze image";
  }
}

function renderSavedResult(record) {
  activeResultId = record.id;
  currentFile = record.file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(record.file);
  preview.src = previewUrl;
  preview.classList.remove("hidden");
  dropzoneHint.classList.add("hidden");
  renderResults(record.data, record.heatmap);
}

function renderResults(data, heatmap) {
  const confidence = Number(data.confidence);
  const pct = confidence * 100;
  const lowConfidence = confidence < 0.3;
  const normalizedLabel = String(data.prediction).trim().toLowerCase();
  resultsHeading.textContent = lowConfidence
    ? "This doesn't look like a clear skin lesion photo"
    : "Analysis result";
  predictionLabel.textContent = lowConfidence ? "" : data.prediction;
  predictionLabel.classList.toggle("hidden", lowConfidence);
  meaningLine.textContent = lowConfidence
    ? "The model couldn't find a spot it's confident about. If this is a skin photo, try a closer, clearer shot."
    : classMeanings[normalizedLabel] || "";
  cautionNote.classList.toggle(
    "hidden",
    lowConfidence || (normalizedLabel !== "actinic keratosis" && normalizedLabel !== "dermatofibroma")
  );
  confidenceBar.classList.toggle("hidden", lowConfidence);
  confidenceText.classList.toggle("hidden", lowConfidence);
  confidenceBadge.classList.toggle("hidden", lowConfidence);
  confidenceFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  confidenceText.textContent = `${pct.toFixed(1)}% confidence`;

  confidenceBadge.classList.remove("medium", "low");
  if (confidence > 0.8) confidenceBadge.textContent = "High confidence";
  else if (confidence > 0.5) {
    confidenceBadge.textContent = "Medium";
    confidenceBadge.classList.add("medium");
  } else {
    confidenceBadge.textContent = "Low";
    confidenceBadge.classList.add("low");
  }

  classList.replaceChildren();
  for (const item of data.all) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = item.label;
    const score = document.createElement("span");
    score.textContent = `${(item.confidence * 100).toFixed(1)}%`;
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, item.confidence * 100))}%`;
    track.append(fill);
    li.append(name, score, track);
    classList.append(li);
  }

  heatmapBox.classList.toggle("hidden", lowConfidence);
  if (heatmap && !lowConfidence) {
    heatmapImg.src = heatmap;
    heatmapImg.classList.remove("hidden");
  } else {
    heatmapImg.removeAttribute("src");
    heatmapImg.classList.add("hidden");
  }
  heatmapLoading.classList.add("hidden");
  resultsCard.classList.remove("hidden");
  resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderRecentResults() {
  recentList.replaceChildren();
  recentSection.classList.toggle("hidden", recentResults.length === 0);

  recentResults.forEach((record) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-item";
    button.classList.toggle("active", record.id === activeResultId);
    button.setAttribute("aria-label", `Show ${record.data.prediction}, ${(record.data.confidence * 100).toFixed(1)}% confidence`);

    const thumbnail = document.createElement("img");
    thumbnail.src = record.imageUrl;
    thumbnail.alt = "";
    const copy = document.createElement("span");
    copy.className = "recent-copy";
    const label = document.createElement("span");
    label.className = "recent-class";
    label.textContent = record.data.prediction;
    const confidence = document.createElement("span");
    confidence.className = "recent-confidence";
    confidence.textContent = `${(record.data.confidence * 100).toFixed(1)}% confidence`;
    const time = document.createElement("time");
    time.className = "recent-time";
    time.dateTime = record.createdAt.toISOString();
    time.textContent = record.createdAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    copy.append(label, confidence, time);
    button.append(thumbnail, copy);
    button.addEventListener("click", () => {
      renderSavedResult(record);
      renderRecentResults();
    });
    recentList.append(button);
  });
}
