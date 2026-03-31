// ── API Base URL ────────────────────────────────────────────────────────
// In production (Render), set window.API_URL via a <script> tag or env injection.
// Falls back to relative path (works with nginx proxy in local Docker).
const API_BASE = window.API_URL || '';

// ── State ──────────────────────────────────────────────────────────────
const state = {
  file: null,
  targetW: 1920,
  targetH: 1080,
  strategy: 'recompose',
  pollingInterval: null,
  currentJobId: null,
};

// ── DOM References ──────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const dropZone    = $('dropZone');
const fileInput   = $('fileInput');
const filePreview = $('filePreview');
const previewThumb = $('previewThumb');
const previewName  = $('previewName');
const previewSize  = $('previewSize');
const btnClearFile = $('btnClearFile');
const btnConvert   = $('btnConvert');
const btnSpinner   = $('btnSpinner');
const statusCard   = $('statusCard');
const statusTitle  = $('statusTitle');
const progressBar  = $('progressBar');
const progressLabel = $('progressLabel');
const badgeStatus  = $('badgeStatus');
const badgeType    = $('badgeType');
const resultArea   = $('resultArea');
const resultPreview = $('resultPreview');
const btnDownload  = $('btnDownload');
const errorArea    = $('errorArea');
const errorMsg     = $('errorMsg');
const dimDisplay   = $('dimDisplay');
const customDims   = $('customDims');
const inputW       = $('inputW');
const inputH       = $('inputH');
const historyBody  = $('historyBody');

// ── Drop Zone ──────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

btnClearFile.addEventListener('click', () => {
  state.file = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  dropZone.classList.remove('hidden');
  btnConvert.disabled = true;
});

function setFile(file) {
  state.file = file;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  previewThumb.innerHTML = '';

  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    previewThumb.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const vid = document.createElement('video');
    vid.src = URL.createObjectURL(file);
    vid.muted = true;
    previewThumb.appendChild(vid);
  }

  dropZone.classList.add('hidden');
  filePreview.classList.remove('hidden');
  btnConvert.disabled = false;
}

// ── Preset Buttons ─────────────────────────────────────────────────────
document.querySelectorAll('.preset-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const w = parseInt(btn.dataset.w);
    const h = parseInt(btn.dataset.h);

    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (w === 0 && h === 0) {
      // Custom
      customDims.classList.remove('hidden');
      state.targetW = parseInt(inputW.value) || 1920;
      state.targetH = parseInt(inputH.value) || 1080;
    } else {
      customDims.classList.add('hidden');
      state.targetW = w;
      state.targetH = h;
    }
    updateDimDisplay();
  });
});

[inputW, inputH].forEach(inp => {
  inp.addEventListener('input', () => {
    state.targetW = parseInt(inputW.value) || 1920;
    state.targetH = parseInt(inputH.value) || 1080;
    updateDimDisplay();
  });
});

function updateDimDisplay() {
  dimDisplay.textContent = `${state.targetW} × ${state.targetH}`;
}

// ── Strategy Cards ──────────────────────────────────────────────────────
document.querySelectorAll('.strategy-card').forEach((card) => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    state.strategy = card.dataset.value;
    card.querySelector('input[type=radio]').checked = true;
  });
});

// ── Convert Button ─────────────────────────────────────────────────────
btnConvert.addEventListener('click', async () => {
  if (!state.file) return;

  setConverting(true);
  resetStatusCard();
  statusCard.classList.remove('hidden');
  statusCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const formData = new FormData();
  formData.append('file', state.file);
  formData.append('target_width', state.targetW);
  formData.append('target_height', state.targetH);
  formData.append('strategy', state.strategy);

  try {
    const res = await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'שגיאה לא ידועה' }));
      throw new Error(err.detail || JSON.stringify(err));
    }
    const job = await res.json();
    state.currentJobId = job.id;
    startPolling(job.id);
  } catch (err) {
    showError(err.message);
    setConverting(false);
  }
});

// ── Polling ─────────────────────────────────────────────────────────────
function startPolling(jobId) {
  stopPolling();
  state.pollingInterval = setInterval(() => pollJob(jobId), 1500);
}

function stopPolling() {
  if (state.pollingInterval) {
    clearInterval(state.pollingInterval);
    state.pollingInterval = null;
  }
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();
    updateStatusUI(job);

    if (job.status === 'done' || job.status === 'failed') {
      stopPolling();
      setConverting(false);
      loadHistory();
    }
  } catch (_) {}
}

// ── Status UI ───────────────────────────────────────────────────────────
const STATUS_LABELS = {
  pending:    'ממתין בתור',
  processing: 'מעבד...',
  done:       'הושלם ✓',
  failed:     'נכשל ✗',
};

const STATUS_BADGE_CLASS = {
  pending:    'badge-pending',
  processing: 'badge-processing',
  done:       'badge-done',
  failed:     'badge-failed',
};

function updateStatusUI(job) {
  statusTitle.textContent = STATUS_LABELS[job.status] || job.status;

  const pct = job.status === 'done' ? 100 : (job.progress || 0);
  progressBar.style.width = `${pct}%`;
  progressLabel.textContent = `${pct}%`;

  badgeStatus.textContent = STATUS_LABELS[job.status];
  badgeStatus.className = `badge ${STATUS_BADGE_CLASS[job.status] || ''}`;
  badgeType.textContent = job.input_type === 'image' ? '🖼 תמונה' : '🎬 סרטון';

  if (job.status === 'done' && job.download_url) {
    resultArea.classList.remove('hidden');
    btnDownload.href = job.download_url;
    btnDownload.download = `converted_${job.target_width}x${job.target_height}_${job.original_filename}`;

    resultPreview.innerHTML = '';
    if (job.input_type === 'image') {
      const img = document.createElement('img');
      img.src = job.download_url;
      img.alt = 'תוצאה';
      resultPreview.appendChild(img);
    } else {
      const vid = document.createElement('video');
      vid.src = job.download_url;
      vid.controls = true;
      vid.style.maxWidth = '100%';
      resultPreview.appendChild(vid);
    }
  }

  if (job.status === 'failed') {
    errorArea.classList.remove('hidden');
    errorMsg.textContent = job.error_message || 'שגיאה לא ידועה. נסה שוב.';
  }
}

function resetStatusCard() {
  statusTitle.textContent = 'מעבד...';
  progressBar.style.width = '0%';
  progressLabel.textContent = '0%';
  badgeStatus.textContent = 'ממתין';
  badgeStatus.className = 'badge badge-pending';
  resultArea.classList.add('hidden');
  errorArea.classList.add('hidden');
  resultPreview.innerHTML = '';
}

function showError(msg) {
  statusCard.classList.remove('hidden');
  statusTitle.textContent = 'שגיאה';
  errorArea.classList.remove('hidden');
  errorMsg.textContent = msg;
  badgeStatus.textContent = 'נכשל ✗';
  badgeStatus.className = 'badge badge-failed';
}

function setConverting(loading) {
  btnConvert.disabled = loading;
  btnSpinner.classList.toggle('hidden', !loading);
  document.querySelector('.btn-label').textContent = loading ? 'מעבד...' : 'המר עכשיו';
}

// ── History ─────────────────────────────────────────────────────────────
$('btnRefreshHistory').addEventListener('click', loadHistory);

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/jobs`);
    if (!res.ok) return;
    const jobs = await res.json();
    renderHistory(jobs);
  } catch (_) {}
}

function renderHistory(jobs) {
  if (jobs.length === 0) {
    historyBody.innerHTML = '<tr><td colspan="6" class="empty-row">אין המרות עדיין</td></tr>';
    return;
  }

  historyBody.innerHTML = jobs.map(job => `
    <tr>
      <td title="${job.original_filename}">${truncate(job.original_filename, 20)}</td>
      <td>${job.input_type === 'image' ? '🖼' : '🎬'}</td>
      <td>${job.target_width}×${job.target_height}</td>
      <td>${strategyLabel(job.strategy)}</td>
      <td><span class="badge ${STATUS_BADGE_CLASS[job.status] || ''}">${STATUS_LABELS[job.status] || job.status}</span></td>
      <td>${job.download_url
        ? `<a href="${job.download_url}" class="btn-text" download>⬇️ הורד</a>`
        : (job.status === 'failed' ? '✗' : '⏳')
      }</td>
    </tr>
  `).join('');
}

function strategyLabel(s) {
  const m = {
    recompose:  '🔬 Recompose',
    fit_blur:   '🌟 Fit+Blur',
    upscale:    '✨ Upscale',
    fit_pad:    '🖼 Fit+Pad',
    smart_crop: '🧠 SmartCrop',
    stretch:    '↔ Stretch',
  };
  return m[s] || s;
}

// ── Utils ────────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max) + '...' : str;
}

// ── Init ─────────────────────────────────────────────────────────────────
loadHistory();
