const taskSelect = document.getElementById("task-select");
const taskHint = document.getElementById("task-hint");
const systemPromptEl = document.getElementById("system-prompt");
const userTextEl = document.getElementById("user-text");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewRow = document.getElementById("preview-row");
const runBtn = document.getElementById("run-btn");
const statusPill = document.getElementById("model-status");
const latencyDisplay = document.getElementById("latency-display");
const runError = document.getElementById("run-error");
const rawOutput = document.getElementById("raw-output");
const parsedOutput = document.getElementById("parsed-output");
const historyBody = document.querySelector("#history-table tbody");
const maxTokensEl = document.getElementById("max-tokens");
const temperatureEl = document.getElementById("temperature");
const enableThinkingEl = document.getElementById("enable-thinking");

let tasks = [];
let selectedFiles = [];
let historyCount = 0;
let modelReady = false;

function taskLabel(taskId) {
  const t = tasks.find(x => x.id === taskId);
  return t ? t.label : taskId;
}

function addHistoryRow({ taskId, timestampSec, elapsedSeconds, promptTokens, completionTokens, tokensPerSecond }) {
  historyCount += 1;
  const when = timestampSec ? new Date(timestampSec * 1000).toLocaleTimeString() : "—";
  const row = document.createElement("tr");
  row.innerHTML = `<td>${historyCount}</td><td>${when}</td><td>${taskLabel(taskId)}</td>` +
    `<td>${fmtLatency(elapsedSeconds)}</td>` +
    `<td>${promptTokens ?? "—"} / ${completionTokens ?? "—"}</td>` +
    `<td>${tokensPerSecond ?? "—"}</td>`;
  historyBody.prepend(row);
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const records = await res.json();
    // API returns oldest-first; addHistoryRow prepends, so replay oldest-first to end
    // up with newest-first display and a correctly incrementing #.
    records.forEach(r => addHistoryRow({
      taskId: r.task_id,
      timestampSec: r.timestamp,
      elapsedSeconds: r.result?.elapsed_seconds,
      promptTokens: r.result?.prompt_tokens,
      completionTokens: r.result?.completion_tokens,
      tokensPerSecond: r.result?.tokens_per_second,
    }));
  } catch (e) {
    console.error("Failed to load run history:", e);
  }
}

async function loadTasks() {
  const res = await fetch("/api/tasks");
  tasks = await res.json();
  taskSelect.innerHTML = tasks.map(t => `<option value="${t.id}">${t.label}</option>`).join("");
  applyTaskPreset(tasks[0].id);
}

function applyTaskPreset(taskId) {
  const t = tasks.find(x => x.id === taskId);
  if (!t) return;
  systemPromptEl.value = t.system_prompt;
  userTextEl.value = t.user_text;
  maxTokensEl.value = t.max_new_tokens || 1024;
  taskHint.textContent = t.expects_json
    ? "This preset asks the model to answer in strict JSON — see 'Parsed JSON' panel after running."
    : "Free-form answer expected — no JSON parsing attempted.";
}

taskSelect.addEventListener("change", () => applyTaskPreset(taskSelect.value));

function renderPreviews() {
  previewRow.innerHTML = "";
  selectedFiles.forEach(f => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    previewRow.appendChild(img);
  });
  updateRunEnabled();
}

function addFiles(fileList) {
  selectedFiles = selectedFiles.concat(Array.from(fileList).filter(f => f.type.startsWith("image/")));
  renderPreviews();
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => addFiles(e.target.files));
["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", e => addFiles(e.dataTransfer.files));

function updateRunEnabled() {
  runBtn.disabled = !modelReady || selectedFiles.length === 0;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    if (s.state === "ready") {
      statusPill.textContent = "model ready";
      statusPill.className = "status-pill status-ready";
      modelReady = true;
      updateRunEnabled();
      return;
    } else if (s.state === "error") {
      statusPill.textContent = "model failed to load — see console";
      statusPill.className = "status-pill status-error";
      console.error("Model load error:", s.error);
      return;
    } else {
      statusPill.textContent = s.state === "loading" ? "loading model… (can take minutes)" : "starting…";
      statusPill.className = "status-pill status-loading";
    }
  } catch (e) {
    statusPill.textContent = "backend unreachable";
    statusPill.className = "status-pill status-error";
  }
  setTimeout(pollStatus, 3000);
}

function fmtLatency(sec) {
  if (sec == null) return "—";
  if (sec < 60) return `${sec.toFixed(2)} s`;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m}m ${s}s`;
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "Running…";
  runError.hidden = true;
  latencyDisplay.textContent = "running…";
  rawOutput.textContent = "";
  parsedOutput.textContent = "";

  const form = new FormData();
  form.append("task_id", taskSelect.value);
  form.append("system_prompt", systemPromptEl.value);
  form.append("user_text", userTextEl.value);
  form.append("max_new_tokens", maxTokensEl.value);
  form.append("temperature", temperatureEl.value);
  form.append("enable_thinking", enableThinkingEl.checked);
  selectedFiles.forEach(f => form.append("images", f));

  try {
    const res = await fetch("/api/run", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) {
      runError.hidden = false;
      runError.textContent = data.error;
      latencyDisplay.textContent = "—";
    } else {
      latencyDisplay.textContent = fmtLatency(data.elapsed_seconds);
      rawOutput.textContent = data.raw_text || "(empty)";
      parsedOutput.textContent = data.parsed_json ? JSON.stringify(data.parsed_json, null, 2) : "(no valid JSON found)";
      addHistoryRow({
        taskId: taskSelect.value,
        timestampSec: Date.now() / 1000,
        elapsedSeconds: data.elapsed_seconds,
        promptTokens: data.prompt_tokens,
        completionTokens: data.completion_tokens,
        tokensPerSecond: data.tokens_per_second,
      });
    }
  } catch (e) {
    runError.hidden = false;
    runError.textContent = String(e);
    latencyDisplay.textContent = "—";
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run";
    updateRunEnabled();
  }
});

async function init() {
  await loadTasks();   // tasks must be loaded first so taskLabel() resolves in history rows
  await loadHistory();
}
init();
pollStatus();
