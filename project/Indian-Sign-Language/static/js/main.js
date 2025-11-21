// ================== Sign Language Detection Control ==================

const video = document.getElementById("video");
const sentenceEl = document.getElementById("sentence");
const currentLetterEl = document.getElementById("currentLetter");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

let detectionRunning = false;

// --- Start Detection ---
startBtn.addEventListener("click", () => {
  fetch("/start", { method: "POST" })
    .then(() => {
      detectionRunning = true;
      video.src = "/video_feed"; // Start live feed
      updateOutputs();
      console.log("✅ Detection started");
    })
    .catch(err => console.error("Error starting detection:", err));
});

// --- Stop Detection ---
stopBtn.addEventListener("click", () => {
  fetch("/stop", { method: "POST" })
    .then(() => {
      detectionRunning = false;
      video.src = "";
      sentenceEl.textContent = "";
      currentLetterEl.textContent = "-";
      console.log("🛑 Detection stopped");
    })
    .catch(err => console.error("Error stopping detection:", err));
});

// --- Clear Sentence ---
clearBtn.addEventListener("click", () => {
  fetch("/clear", { method: "POST" })
    .then(() => {
      sentenceEl.textContent = "";
      currentLetterEl.textContent = "-";
      console.log("🧹 Sentence cleared");
    })
    .catch(err => console.error("Error clearing sentence:", err));
});

// --- Update Sentence & Current Letter ---
function updateOutputs() {
  if (!detectionRunning) return;

  fetch("/get_sentence")
    .then(response => response.json())
    .then(data => {
      sentenceEl.textContent = data.sentence || "";
      currentLetterEl.textContent = data.current || "-";
      if (detectionRunning) {
        setTimeout(updateOutputs, 1000);
      }
    })
    .catch(err => console.error("Error fetching data:", err));
}
