const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const fileSelected = document.getElementById('fileSelected');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const progressBar = document.getElementById('progressBar');

let selectedFile = null;
let selectedSpeed = 1.0;

const stepMap = {
  "step:1": { el: "step1", progress: 20 },
  "step:2": { el: "step2", progress: 40 },
  "step:3": { el: "step3", progress: 60 },
  "step:4": { el: "step4", progress: 80 },
  "step:5": { el: "step5", progress: 95 },
};

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) selectFile(e.target.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  uploadZone.style.display = 'none';
  fileSelected.style.display = 'flex';
}

function changeVideo() {
  selectedFile = null;
  fileInput.value = '';
  uploadZone.style.display = 'flex';
  fileSelected.style.display = 'none';
}

function setSpeed(btn) {
  document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedSpeed = parseFloat(btn.dataset.speed);
}

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});

function markStepDone(stepEl) {
  stepEl.classList.remove('active');
  stepEl.classList.add('done');
}

function markStepActive(stepEl) {
  stepEl.classList.add('active');
}

async function startAnalysis() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  document.querySelector('.hero').style.display = 'none';
  progressSection.style.display = 'block';
  resultsSection.style.display = 'none';

  // Reset steps
  for (let i = 1; i <= 5; i++) {
    const step = document.getElementById(`step${i}`);
    step.classList.remove('active', 'done');
  }
  progressBar.style.width = '5%';

  // Upload video first
  const formData = new FormData();
  formData.append('video', selectedFile);
  formData.append('speed', selectedSpeed);

  let sessionId;
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) { showError(data.error); resetUI(); return; }
    sessionId = data.session_id;
  } catch (err) {
    showError('Upload failed. Please try again.');
    resetUI();
    return;
  }

  // Connect to SSE stream
  const evtSource = new EventSource(`/stream/${sessionId}`);

  evtSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'progress') {
      const key = msg.data.split(':').slice(0, 2).join(':');
      const stepInfo = stepMap[key];
      if (stepInfo) {
        // Mark all previous steps done
        Object.entries(stepMap).forEach(([k, info]) => {
          const stepEl = document.getElementById(info.el);
          if (info.progress < stepInfo.progress) {
            markStepDone(stepEl);
          }
        });
        markStepActive(document.getElementById(stepInfo.el));
        progressBar.style.width = `${stepInfo.progress}%`;
      }
    }

    if (msg.type === 'done') {
      evtSource.close();
      // Mark all done
      for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active');
        step.classList.add('done');
      }
      progressBar.style.width = '100%';

      // Fetch results
      fetch(`/results_data/${msg.session_id}`)
        .then(r => r.json())
        .then(data => {
          if (data.error) { showError(data.error); resetUI(); return; }
          setTimeout(() => {
            progressSection.style.display = 'none';
            showResults(data, msg.session_id);
          }, 500);
        })
        .catch(() => { showError('Failed to load results.'); resetUI(); });
    }

    if (msg.type === 'error') {
      evtSource.close();
      showError(msg.message || 'Analysis failed.');
      resetUI();
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    showError('Connection lost. Please try again.');
    resetUI();
  };
}

function showResults(data, sid) {
  document.getElementById('loadImg').src = `/results/${sid}/load_frame.jpg`;
  document.getElementById('releaseImg').src = `/results/${sid}/analyzed_frame.jpg`;
  document.getElementById('followthroughImg').src = `/results/${sid}/followthrough_frame.jpg`;

  document.getElementById('releaseMetrics').innerHTML = buildReleaseMetrics(data.release_metrics || {});
  document.getElementById('loadMetrics').innerHTML = buildLoadMetrics(data.load_metrics || {});
  document.getElementById('followthroughMetrics').innerHTML = buildFollowthroughMetrics(data.followthrough_metrics || {});

  const coaching = data.coaching || 'No coaching report generated. Please set your Groq API key.';
  document.getElementById('coachingText').textContent = coaching;

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function metricCard(label, value, unit, status) {
  const cls = status === 'good' ? 'good' : status === 'bad' ? 'bad' : 'warn';
  const badge = status === 'good' ? '<span class="metric-badge good">Good</span>' :
                status === 'bad' ? '<span class="metric-badge bad">Needs Work</span>' : '';
  return `<div class="metric-card">
    <div class="metric-label">${label}</div>
    <div class="metric-value ${cls}">${value}<small style="font-size:1rem">${unit}</small></div>
    ${badge}
  </div>`;
}

function boolCard(label, value, goodWhen) {
  const isGood = value === goodWhen;
  const display = value === null ? 'N/A' : value ? 'YES' : 'NO';
  return metricCard(label, display, '', value === null ? '' : isGood ? 'good' : 'bad');
}

function buildReleaseMetrics(rm) {
  let html = '';
  if (rm.elbow_angle_deg != null) html += metricCard('Elbow Angle', rm.elbow_angle_deg.toFixed(1), '°', rm.elbow_angle_deg >= 150 ? 'good' : 'bad');
  if (rm.knee_angle_deg != null) html += metricCard('Knee Angle', rm.knee_angle_deg.toFixed(1), '°', rm.knee_angle_deg >= 160 ? 'good' : 'warn');
  if (rm.wrist_above_shoulder != null) html += boolCard('Wrist Above Shoulder', rm.wrist_above_shoulder, true);
  if (rm.elbow_offset_px != null && rm.flare_threshold_px != null) {
    const flared = Math.abs(rm.elbow_offset_px) > rm.flare_threshold_px;
    html += metricCard('Elbow Alignment', flared ? 'FLARED' : 'STACKED', '', flared ? 'bad' : 'good');
  }
  return html || '<p style="color:var(--text-dim)">No release data available.</p>';
}

function buildLoadMetrics(lm) {
  let html = '';
  if (lm.knee_angle_left != null) html += metricCard('Left Knee Bend', lm.knee_angle_left.toFixed(1), '°', lm.knee_angle_left <= 160 ? 'good' : 'warn');
  if (lm.knee_angle_right != null) html += metricCard('Right Knee Bend', lm.knee_angle_right.toFixed(1), '°', lm.knee_angle_right <= 160 ? 'good' : 'warn');
  if (lm.elbow_angle_deg != null) html += metricCard('Elbow Angle', lm.elbow_angle_deg.toFixed(1), '°', lm.elbow_angle_deg <= 90 ? 'good' : 'warn');
  if (lm.hip_square != null) html += boolCard('Hips Square', lm.hip_square, true);
  if (lm.balance_ok != null) html += boolCard('Body Balanced', lm.balance_ok, true);
  return html || '<p style="color:var(--text-dim)">No load data available.</p>';
}

function buildFollowthroughMetrics(fm) {
  let html = '';
  if (fm.wrist_snapped != null) html += boolCard('Wrist Snap', fm.wrist_snapped, true);
  if (fm.elbow_angle_deg != null) html += metricCard('Elbow Angle', fm.elbow_angle_deg.toFixed(1), '°', fm.elbow_angle_deg >= 160 ? 'good' : 'warn');
  if (fm.balance_ok != null) html += boolCard('Body Balanced', fm.balance_ok, true);
  return html || '<p style="color:var(--text-dim)">No follow through data available.</p>';
}

function showTab(name, e) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.metrics-panel').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(`panel-${name}`).classList.add('active');
}

function showError(msg) {
  const toast = document.getElementById('errorToast');
  document.getElementById('errorMsg').textContent = msg;
  toast.style.display = 'flex';
  setTimeout(() => toast.style.display = 'none', 5000);
}

function resetUI() {
  selectedFile = null;
  fileInput.value = '';
  analyzeBtn.disabled = false;
  uploadZone.style.display = 'flex';
  fileSelected.style.display = 'none';
  progressSection.style.display = 'none';
  resultsSection.style.display = 'none';
  document.querySelector('.hero').style.display = 'grid';
  progressBar.style.width = '0%';
  for (let i = 1; i <= 5; i++) {
    document.getElementById(`step${i}`).classList.remove('active', 'done');
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}