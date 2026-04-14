const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const fileSelected = document.getElementById('fileSelected');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const progressBar = document.getElementById('progressBar');

let selectedFile = null;

// File selection
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) selectFile(e.target.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  uploadZone.style.display = 'none';
  fileSelected.style.display = 'flex';
}

// Drag and drop
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});

// Progress animation
const stepMessages = [
  'Locating basketball',
  'Detecting body pose',
  'Finding release point',
  'Calculating angles',
  'Generating AI coaching'
];

let stepInterval = null;
let currentStep = 0;

function startProgressAnimation() {
  currentStep = 0;
  progressBar.style.width = '0%';
  stepInterval = setInterval(() => {
    if (currentStep < stepMessages.length) {
      // Mark previous done
      if (currentStep > 0) {
        const prev = document.getElementById(`step${currentStep}`);
        prev.classList.remove('active');
        prev.classList.add('done');
      }
      // Mark current active
      const curr = document.getElementById(`step${currentStep + 1}`);
      curr.classList.add('active');
      progressBar.style.width = `${((currentStep + 1) / stepMessages.length) * 80}%`;
      currentStep++;
    }
  }, 8000 / stepMessages.length);
}

function finishProgress() {
  clearInterval(stepInterval);
  // Mark all done
  for (let i = 1; i <= 5; i++) {
    const step = document.getElementById(`step${i}`);
    step.classList.remove('active');
    step.classList.add('done');
  }
  progressBar.style.width = '100%';
}

// Main analysis
async function startAnalysis() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  document.querySelector('.hero').style.display = 'none';
  progressSection.style.display = 'block';
  resultsSection.style.display = 'none';

  startProgressAnimation();

  const formData = new FormData();
  formData.append('video', selectedFile);

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    finishProgress();

    if (data.error) {
      showError(data.error);
      resetUI();
      return;
    }

    setTimeout(() => {
      progressSection.style.display = 'none';
      showResults(data);
    }, 600);

  } catch (err) {
    finishProgress();
    showError('Network error. Please try again.');
    resetUI();
  }
}

function showResults(data) {
  const sid = data.session_id;

  // Images
  document.getElementById('loadImg').src = `/results/${sid}/load_frame.jpg`;
  document.getElementById('releaseImg').src = `/results/${sid}/analyzed_frame.jpg`;
  document.getElementById('followthroughImg').src = `/results/${sid}/followthrough_frame.jpg`;

  // Release metrics
  const rm = data.release_metrics || {};
  document.getElementById('releaseMetrics').innerHTML = buildReleaseMetrics(rm);

  // Load metrics
  const lm = data.load_metrics || {};
  document.getElementById('loadMetrics').innerHTML = buildLoadMetrics(lm);

  // Follow through metrics
  const fm = data.followthrough_metrics || {};
  document.getElementById('followthroughMetrics').innerHTML = buildFollowthroughMetrics(fm);

  // Coaching
  const coaching = data.coaching || 'No coaching report generated. Please set your Groq API key.';
  document.getElementById('coachingText').textContent = coaching;

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function metricCard(label, value, unit, status) {
  const cls = status === 'good' ? 'good' : status === 'bad' ? 'bad' : 'warn';
  const badge = status === 'good' ? '<span class="metric-badge good">Good</span>' :
                status === 'bad' ? '<span class="metric-badge bad">Needs Work</span>' : '';
  return `
    <div class="metric-card">
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
  if (rm.elbow_angle_deg != null) {
    html += metricCard('Elbow Angle', rm.elbow_angle_deg.toFixed(1), '°', rm.elbow_angle_deg >= 150 ? 'good' : 'bad');
  }
  if (rm.knee_angle_deg != null) {
    html += metricCard('Knee Angle', rm.knee_angle_deg.toFixed(1), '°', rm.knee_angle_deg >= 160 ? 'good' : 'warn');
  }
  if (rm.wrist_above_shoulder != null) {
    html += boolCard('Wrist Above Shoulder', rm.wrist_above_shoulder, true);
  }
  if (rm.elbow_offset_px != null && rm.flare_threshold_px != null) {
    const flared = Math.abs(rm.elbow_offset_px) > rm.flare_threshold_px;
    html += metricCard('Elbow Alignment', flared ? 'FLARED' : 'STACKED', '', flared ? 'bad' : 'good');
  }
  return html || '<p style="color:var(--text-dim)">No release data available.</p>';
}

function buildLoadMetrics(lm) {
  let html = '';
  if (lm.knee_angle_left != null) {
    html += metricCard('Left Knee Bend', lm.knee_angle_left.toFixed(1), '°', lm.knee_angle_left <= 160 ? 'good' : 'warn');
  }
  if (lm.knee_angle_right != null) {
    html += metricCard('Right Knee Bend', lm.knee_angle_right.toFixed(1), '°', lm.knee_angle_right <= 160 ? 'good' : 'warn');
  }
  if (lm.elbow_angle_deg != null) {
    html += metricCard('Elbow Angle', lm.elbow_angle_deg.toFixed(1), '°', lm.elbow_angle_deg <= 90 ? 'good' : 'warn');
  }
  if (lm.hip_square != null) {
    html += boolCard('Hips Square', lm.hip_square, true);
  }
  if (lm.balance_ok != null) {
    html += boolCard('Body Balanced', lm.balance_ok, true);
  }
  return html || '<p style="color:var(--text-dim)">No load data available.</p>';
}

function buildFollowthroughMetrics(fm) {
  let html = '';
  if (fm.wrist_snapped != null) {
    html += boolCard('Wrist Snap', fm.wrist_snapped, true);
  }
  if (fm.elbow_angle_deg != null) {
    html += metricCard('Elbow Angle', fm.elbow_angle_deg.toFixed(1), '°', fm.elbow_angle_deg >= 160 ? 'good' : 'warn');
  }
  if (fm.balance_ok != null) {
    html += boolCard('Body Balanced', fm.balance_ok, true);
  }
  return html || '<p style="color:var(--text-dim)">No follow through data available.</p>';
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.metrics-panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(`panel-${name}`).classList.add('active');
}

function showError(msg) {
  const toast = document.getElementById('errorToast');
  document.getElementById('errorMsg').textContent = msg;
  toast.style.display = 'flex';
  setTimeout(() => toast.style.display = 'none', 5000);
}

function resetUI() {
  clearInterval(stepInterval);
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
    const step = document.getElementById(`step${i}`);
    step.classList.remove('active', 'done');
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}