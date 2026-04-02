// ── API Configuration ────────────────────────────────────────────────────────
// For Vercel + Colab deployment: set to your ngrok static domain
const NGROK_DOMAIN = 'https://nickelic-deserved-nilda.ngrok-free.dev';

function getBaseUrl() {
    const isLocal = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost';
    return isLocal ? 'http://localhost:8000' : NGROK_DOMAIN;
}

// Wrapper around fetch that adds the ngrok-skip-browser-warning header
// (required for free ngrok domains, otherwise ngrok returns an interstitial HTML page)
function apiFetch(url, options = {}) {
    options.headers = {
        ...options.headers,
        'ngrok-skip-browser-warning': 'true',
    };
    return fetch(url, options);
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const promptInput    = document.getElementById('prompt-input');
const sendButton     = document.getElementById('send-button');
const chatContainer  = document.getElementById('chat-container');
const statusText     = document.getElementById('status-text');
const dropdownTrigger = document.getElementById('dropdown-trigger');
const dropdownPanel  = document.getElementById('dropdown-panel');
const dropdownLabel  = document.getElementById('dropdown-label');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedMode    = 'full';
let selectedModelId = 'meta-llama/Llama-3.1-8B-Instruct';

// ── Hallucination score history ──────────────────────────────────────────────
let scoreHistory = [];

function restoreScoreHistory() {
    try {
        const saved = sessionStorage.getItem('se_score_history');
        if (saved) scoreHistory = JSON.parse(saved);
    } catch (_) { scoreHistory = []; }
}

function persistScoreHistory() {
    sessionStorage.setItem('se_score_history', JSON.stringify(scoreHistory));
}

function recordScore(question, combinedRisk, mode) {
    scoreHistory.push({
        q: question.length > 40 ? question.substring(0, 40) + '...' : question,
        risk: combinedRisk,
        conf: 1.0 - combinedRisk,
        mode: mode,
        ts: Date.now(),
    });
    persistScoreHistory();
    renderScoreChart();
}

function renderScoreChart() {
    const chartContainer = document.getElementById('score-chart-container');
    const chartCanvas = document.getElementById('score-chart');
    if (!chartContainer || !chartCanvas) return;
    if (scoreHistory.length === 0) {
        chartContainer.classList.add('empty');
        chartCanvas.innerHTML = '<div class="chart-empty">No scores yet. Ask questions to see the trend.</div>';
        return;
    }
    chartContainer.classList.remove('empty');

    const W = chartCanvas.clientWidth || 480;
    const H = 140;
    const pad = { top: 20, right: 16, bottom: 28, left: 40 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const n = scoreHistory.length;

    // Build points
    const points = scoreHistory.map((s, i) => ({
        x: pad.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW),
        y: pad.top + (1 - s.conf) * plotH,
        conf: s.conf,
        risk: s.risk,
        q: s.q,
        mode: s.mode,
    }));

    // Confidence zone backgrounds
    const yHigh = pad.top;
    const yMedTop = pad.top + 0.35 * plotH;
    const yMedBot = pad.top + 0.65 * plotH;
    const yLow = pad.top + plotH;

    let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    // Zone fills
    svg += `<rect x="${pad.left}" y="${yHigh}" width="${plotW}" height="${yMedTop - yHigh}" fill="rgba(52,199,89,0.07)"/>`;
    svg += `<rect x="${pad.left}" y="${yMedTop}" width="${plotW}" height="${yMedBot - yMedTop}" fill="rgba(255,159,10,0.07)"/>`;
    svg += `<rect x="${pad.left}" y="${yMedBot}" width="${plotW}" height="${yLow - yMedBot}" fill="rgba(255,69,58,0.07)"/>`;
    // Zone labels
    svg += `<text x="${pad.left - 4}" y="${(yHigh + yMedTop) / 2 + 3}" text-anchor="end" font-size="9" fill="#34c759" opacity="0.8">HIGH</text>`;
    svg += `<text x="${pad.left - 4}" y="${(yMedTop + yMedBot) / 2 + 3}" text-anchor="end" font-size="9" fill="#ff9f0a" opacity="0.8">MED</text>`;
    svg += `<text x="${pad.left - 4}" y="${(yMedBot + yLow) / 2 + 3}" text-anchor="end" font-size="9" fill="#ff453a" opacity="0.8">LOW</text>`;
    // Grid lines
    svg += `<line x1="${pad.left}" y1="${yMedTop}" x2="${pad.left + plotW}" y2="${yMedTop}" stroke="rgba(0,0,0,0.08)" stroke-dasharray="3,3"/>`;
    svg += `<line x1="${pad.left}" y1="${yMedBot}" x2="${pad.left + plotW}" y2="${yMedBot}" stroke="rgba(0,0,0,0.08)" stroke-dasharray="3,3"/>`;
    // Axes
    svg += `<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${yLow}" stroke="rgba(0,0,0,0.12)"/>`;
    svg += `<line x1="${pad.left}" y1="${yLow}" x2="${pad.left + plotW}" y2="${yLow}" stroke="rgba(0,0,0,0.12)"/>`;

    // Line path
    if (n > 1) {
        const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
        svg += `<path d="${pathD}" fill="none" stroke="#0A84FF" stroke-width="2" stroke-linejoin="round"/>`;
    }

    // Dots with tooltips
    points.forEach((p, i) => {
        const color = p.conf >= 0.65 ? '#34c759' : p.conf >= 0.35 ? '#ff9f0a' : '#ff453a';
        svg += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="${color}" stroke="white" stroke-width="1.5">`;
        svg += `<title>Q${i + 1}: ${p.q}\nConfidence: ${(p.conf * 100).toFixed(1)}%\nMode: ${p.mode}</title>`;
        svg += `</circle>`;
    });

    // X-axis labels
    points.forEach((p, i) => {
        if (n <= 15 || i % Math.ceil(n / 10) === 0 || i === n - 1) {
            svg += `<text x="${p.x.toFixed(1)}" y="${H - 4}" text-anchor="middle" font-size="10" fill="rgba(0,0,0,0.4)">Q${i + 1}</text>`;
        }
    });

    svg += '</svg>';
    chartCanvas.innerHTML = svg;
}

restoreScoreHistory();

const loadingOverlay   = document.getElementById('loading-overlay');
const loadingModelName = document.getElementById('loading-model-name');

// ── Model readiness polling ──────────────────────────────────────────────────
function showLoadingOverlay(modelLabel) {
    loadingModelName.textContent = modelLabel || 'Initializing...';
    loadingOverlay.classList.remove('hidden');
    promptInput.disabled = true;
    sendButton.disabled = true;
}

function hideLoadingOverlay() {
    loadingOverlay.classList.add('hidden');
    promptInput.disabled = false;
    sendButton.disabled = false;
    sendButton.style.opacity = promptInput.value.trim() ? '1' : '0.5';
}

async function pollBackendStatus() {
    const baseUrl = getBaseUrl();

    const poll = async () => {
        try {
            const res = await apiFetch(`${baseUrl}/status`);
            const data = await res.json();
            if (data.ready) {
                hideLoadingOverlay();
                return;
            }
            // Still loading — update label
            loadingModelName.textContent = data.loading_model_id || 'Loading...';
        } catch (e) {
            // Server not reachable yet
            loadingModelName.textContent = 'Waiting for server...';
        }
        setTimeout(poll, 2000);
    };
    poll();
}

// Start polling on page load
showLoadingOverlay('Connecting to server...');
pollBackendStatus();

// ── Preferences (localStorage) ───────────────────────────────────────────────
function savePrefs() {
    localStorage.setItem('se_mode',        selectedMode);
    localStorage.setItem('se_model_id',    selectedModelId);
    localStorage.setItem('se_model_label', dropdownLabel.textContent);
}

function loadPrefs() {
    const savedMode  = localStorage.getItem('se_mode');
    const savedId    = localStorage.getItem('se_model_id');
    const savedLabel = localStorage.getItem('se_model_label');

    if (savedMode)  selectedMode    = savedMode;
    if (savedId)    selectedModelId = savedId;
    if (savedLabel) dropdownLabel.textContent = savedLabel;

    document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === selectedMode));
    document.querySelectorAll('.dropdown-item').forEach(item =>
        item.classList.toggle('active', item.dataset.modelId === selectedModelId));
}

// ── Message history (sessionStorage) ─────────────────────────────────────────
function persistMessages() {
    const msgs = [...chatContainer.querySelectorAll('.message')]
        .filter(m => m.id !== 'welcome-message')   // skip welcome — it's already in the HTML
        .map(m => m.outerHTML);
    sessionStorage.setItem('se_chat', JSON.stringify(msgs));
}

function restoreMessages() {
    const saved = sessionStorage.getItem('se_chat');
    if (!saved) return;
    try {
        JSON.parse(saved).forEach(html => {
            const tmp = document.createElement('div');
            tmp.innerHTML = html;   // safe: restoring our own previously-sanitised HTML
            if (tmp.firstChild) chatContainer.appendChild(tmp.firstChild);
        });
        scrollToBottom();
    } catch (_) {
        sessionStorage.removeItem('se_chat');
    }
}

// ── Scroll ────────────────────────────────────────────────────────────────────
function scrollToBottom() {
    // requestAnimationFrame ensures scroll fires after the browser has painted
    // the new DOM node — avoids the race condition with multiple async scroll calls
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

// ── Dropdown ──────────────────────────────────────────────────────────────────
dropdownTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = dropdownPanel.classList.contains('open');
    dropdownTrigger.classList.toggle('open', !isOpen);
    dropdownPanel.classList.toggle('open', !isOpen);
});

document.addEventListener('click', () => {
    dropdownTrigger.classList.remove('open');
    dropdownPanel.classList.remove('open');
});

document.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', async (e) => {
        e.stopPropagation();
        const newModelId = item.dataset.modelId;
        const newLabel = item.dataset.label;

        dropdownLabel.textContent = newLabel;
        document.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        dropdownTrigger.classList.remove('open');
        dropdownPanel.classList.remove('open');

        // If same model, just update prefs
        if (newModelId === selectedModelId) {
            savePrefs();
            return;
        }

        selectedModelId = newModelId;
        savePrefs();

        // Trigger model switch on backend with loading overlay
        showLoadingOverlay(newModelId);
        const baseUrl = getBaseUrl();
        try {
            const res = await apiFetch(`${baseUrl}/switch_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: newModelId }),
            });
            if (!res.ok) throw new Error(`Server error ${res.status}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            hideLoadingOverlay();
            statusText.textContent = `Switched to ${newLabel}`;
        } catch (err) {
            hideLoadingOverlay();
            statusText.textContent = `Model switch failed: ${err.message}`;
            statusText.style.color = 'var(--conf-low-text)';
        }
    });
});

// ── Mode selector ─────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        selectedMode = btn.dataset.mode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        savePrefs();
    });
});

// ── Score history chart modal ─────────────────────────────────────────────────
const chartOverlay = document.getElementById('chart-overlay');
document.getElementById('chart-btn').addEventListener('click', () => {
    renderScoreChart();
    chartOverlay.classList.add('open');
});
document.getElementById('chart-close').addEventListener('click', () => {
    chartOverlay.classList.remove('open');
});
chartOverlay.addEventListener('click', (e) => {
    if (e.target === chartOverlay) chartOverlay.classList.remove('open');
});

// ── Metrics guide modal ──────────────────────────────────────────────────────
const guideOverlay = document.getElementById('guide-overlay');
document.getElementById('guide-btn').addEventListener('click', () => {
    guideOverlay.classList.add('open');
});
document.getElementById('guide-close').addEventListener('click', () => {
    guideOverlay.classList.remove('open');
});
guideOverlay.addEventListener('click', (e) => {
    if (e.target === guideOverlay) guideOverlay.classList.remove('open');
});

// ── Clear chat ────────────────────────────────────────────────────────────────
document.getElementById('clear-btn').addEventListener('click', () => {
    // Keep only the welcome message (first child)
    const messages = chatContainer.querySelectorAll('.message');
    messages.forEach((m, i) => { if (i > 0) m.remove(); });
    sessionStorage.removeItem('se_chat');
    scoreHistory = [];
    sessionStorage.removeItem('se_score_history');
    renderScoreChart();
});

// ── Textarea auto-resize ──────────────────────────────────────────────────────
promptInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
    sendButton.style.opacity = this.value.trim() ? '1' : '0.5';
});

promptInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener('click', sendMessage);
sendButton.style.opacity = '0.5';

// ── Message rendering ─────────────────────────────────────────────────────────
function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-enter ${isUser ? 'user-message' : 'system-message'}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (isUser) {
        // User input: textContent only — never pass user text through innerHTML
        bubble.textContent = text;
    } else {
        // AI response: HTML-escape first, then apply safe markdown (bold + newlines)
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        const formatted = escaped
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        bubble.innerHTML = formatted;
    }

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function addMessageWithSentenceScores(text, sentenceScores) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-enter system-message';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (!sentenceScores || sentenceScores.length === 0) {
        // Fallback: plain rendering (same as addMessage for AI)
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        const formatted = escaped
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        bubble.innerHTML = formatted;
    } else {
        // Sentence-highlighted rendering with whitespace recovery
        // Build sentence positions in original text
        const sentSpans = [];
        let searchPos = 0;
        sentenceScores.forEach(s => {
            const foundIdx = text.indexOf(s.text, searchPos);
            const start = foundIdx >= 0 ? foundIdx : searchPos;
            const end = start + s.text.length;
            sentSpans.push({ start, end });
            searchPos = end;
        });

        sentenceScores.forEach((s, idx) => {
            // Insert original whitespace/newlines between sentences
            if (idx > 0) {
                const gap = text.substring(sentSpans[idx - 1].end, sentSpans[idx].start);
                if (gap.includes('\n')) {
                    bubble.appendChild(document.createElement('br'));
                    if ((gap.match(/\n/g) || []).length >= 2) {
                        bubble.appendChild(document.createElement('br'));
                    }
                } else {
                    bubble.appendChild(document.createTextNode(gap || ' '));
                }
            }

            const span = document.createElement('span');
            span.className = 'sentence-span';

            if (s.is_claim === false) {
                // Non-claim sentence: dimmed, no risk highlighting
                span.classList.add('sentence-non-claim');
                span.title = 'Non-claim sentence (not scored)';
            } else {
                if (s.level === 'medium') {
                    span.classList.add('sentence-medium');
                } else if (s.level === 'low') {
                    span.classList.add('sentence-low');
                }

                // Build tooltip with both logit confidence and probe risk
                const parts = [];
                if (s.confidence != null) parts.push(`Logit confidence: ${(s.confidence * 100).toFixed(1)}%`);
                if (s.energy_risk != null) parts.push(`Energy risk: ${(s.energy_risk * 100).toFixed(1)}%`);
                if (s.entropy_risk != null) parts.push(`Entropy risk: ${(s.entropy_risk * 100).toFixed(1)}%`);
                if (s.probe_risk != null) parts.push(`Combined risk: ${(s.probe_risk * 100).toFixed(1)}%`);
                if (parts.length) span.title = parts.join(' | ');
            }

            // HTML-escape the sentence text, apply bold markdown
            const escaped = s.text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            const formatted = escaped
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            span.innerHTML = formatted;

            bubble.appendChild(span);
        });
    }

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-enter system-message typing-msg';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble typing-indicator';
    bubble.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    msgDiv.appendChild(bubble);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function appendConfidenceBadge(messageEl, { score, level, clusters = null, mode = null }) {
    const percentage = (score * 100).toFixed(1);

    let badgeClass, iconLabel;
    if (level === 'high') {
        badgeClass = 'conf-high';
        iconLabel  = 'High Confidence';
    } else if (level === 'medium') {
        badgeClass = 'conf-medium';
        iconLabel  = 'Medium Confidence';
    } else {
        badgeClass = 'conf-low';
        iconLabel  = 'Hallucination Risk';
    }

    const clusterInfo = clusters !== null
        ? `<span style="opacity:0.6;font-weight:normal;text-transform:none;margin-left:4px">• ${clusters} Semantic Clusters</span>`
        : '';
    const modeInfo = mode
        ? `<span style="opacity:0.6;font-weight:normal;text-transform:none;margin-left:4px">• ${mode}</span>`
        : '';

    const badge = document.createElement('div');
    badge.className = `confidence-badge ${badgeClass}`;
    badge.innerHTML = `<div class="dot"></div><span>${iconLabel} (${percentage}%)</span>${clusterInfo}${modeInfo}`;
    messageEl.appendChild(badge);
}

function appendTimer(messageEl, seconds) {
    const timerEl = document.createElement('div');
    timerEl.className = 'response-timer';
    timerEl.textContent = `⏱ Generated in ${seconds}s`;
    messageEl.appendChild(timerEl);
}

function buildMetricRow(label, displayVal, pct) {
    const row = document.createElement('div');
    row.className = 'metric-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'metric-label';
    labelEl.textContent = label;

    const valueEl = document.createElement('span');
    valueEl.className = 'metric-value';
    valueEl.textContent = displayVal;

    row.appendChild(labelEl);
    row.appendChild(valueEl);

    if (pct !== null) {
        const bar = document.createElement('div');
        bar.className = 'metric-bar';
        const fill = document.createElement('div');
        fill.className = 'metric-fill';
        fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
        bar.appendChild(fill);
        row.appendChild(bar);
    }

    return row;
}

function appendMetricsPanel(messageEl, metricsData) {
    const toggle = document.createElement('div');
    toggle.className = 'metrics-toggle';
    toggle.textContent = '▸ Details';

    const panel = document.createElement('div');
    panel.className = 'metrics-panel';
    panel.hidden = true;

    if (metricsData.type === 'probe') {
        // Probe modes (SLT/TBG): show only sentence avg confidence
        if (metricsData.sentence_avg_confidence != null) {
            const avgConf = metricsData.sentence_avg_confidence;
            panel.appendChild(buildMetricRow('Sentence avg conf', (avgConf * 100).toFixed(1) + '%', avgConf * 100));
        } else {
            // Nothing to show (e.g. single-sentence answer) — skip Details entirely
            return;
        }
    } else {
        // Full SE: cluster breakdown + sentence avg conf
        panel.appendChild(buildMetricRow('Clusters found', String(metricsData.clusters_found), null));
        (metricsData.energies || []).forEach((e, i) => {
            panel.appendChild(buildMetricRow(`Cluster ${i + 1} energy`, e.toFixed(3), e * 100));
        });
        if (metricsData.sentence_avg_confidence != null) {
            const avgConf = metricsData.sentence_avg_confidence;
            panel.appendChild(buildMetricRow('Sentence avg conf', (avgConf * 100).toFixed(1) + '%', avgConf * 100));
        }
    }

    toggle.addEventListener('click', () => {
        const isOpen = !panel.hidden;
        panel.hidden = isOpen;
        toggle.textContent = isOpen ? '▸ Details' : '▾ Details';
    });

    messageEl.appendChild(toggle);
    messageEl.appendChild(panel);
}

function appendSentenceAnalysis(messageEl, sentenceScores) {
    if (!sentenceScores || sentenceScores.length === 0) return;

    const toggle = document.createElement('div');
    toggle.className = 'metrics-toggle';
    toggle.textContent = '▸ Sentence Analysis';

    const panel = document.createElement('div');
    panel.className = 'metrics-panel sentence-analysis-panel';
    panel.hidden = true;

    sentenceScores.forEach((s, idx) => {
        const row = document.createElement('div');
        row.className = 'sentence-row';

        const indexBadge = document.createElement('span');
        indexBadge.className = 'sentence-index';
        indexBadge.textContent = `S${idx + 1}`;

        const textEl = document.createElement('span');
        textEl.className = 'sentence-text';
        const maxLen = 60;
        textEl.textContent = s.text.length > maxLen
            ? s.text.substring(0, maxLen) + '...'
            : s.text;
        textEl.title = s.text;

        // Non-claim sentences: show SKIP badge, no confidence/bar
        if (s.is_claim === false) {
            textEl.style.opacity = '0.5';

            const valueEl = document.createElement('span');
            valueEl.className = 'sentence-conf-value';
            valueEl.textContent = '—';
            valueEl.style.color = 'var(--text-secondary)';

            const bar = document.createElement('div');
            bar.className = 'sentence-conf-bar';

            const levelEl = document.createElement('span');
            levelEl.className = 'sentence-level sentence-level-none';
            levelEl.textContent = 'SKIP';

            row.appendChild(indexBadge);
            row.appendChild(textEl);
            row.appendChild(valueEl);
            row.appendChild(bar);
            row.appendChild(levelEl);
            panel.appendChild(row);
            return;
        }

        // Use probe risk as primary display when available, logit confidence as fallback
        const hasProbe = s.probe_risk != null;
        const displayConf = hasProbe ? (1.0 - s.probe_risk) : s.confidence;

        const valueEl = document.createElement('span');
        valueEl.className = 'sentence-conf-value';
        valueEl.textContent = displayConf != null
            ? (displayConf * 100).toFixed(0) + '%'
            : 'N/A';
        valueEl.title = hasProbe
            ? `Combined: ${(displayConf * 100).toFixed(1)}% | Energy: ${((1-(s.energy_risk||0))*100).toFixed(1)}% | Entropy: ${((1-(s.entropy_risk||0))*100).toFixed(1)}%`
            : `Logit confidence: ${((s.confidence || 0) * 100).toFixed(1)}%`;

        const bar = document.createElement('div');
        bar.className = 'sentence-conf-bar';
        const fill = document.createElement('div');
        fill.className = 'sentence-conf-fill';
        if (s.level === 'high') fill.classList.add('fill-high');
        else if (s.level === 'medium') fill.classList.add('fill-medium');
        else fill.classList.add('fill-low');
        fill.style.width = displayConf != null
            ? `${Math.min(100, Math.max(0, displayConf * 100))}%`
            : '0%';
        bar.appendChild(fill);

        const levelEl = document.createElement('span');
        levelEl.className = `sentence-level sentence-level-${s.level}`;
        levelEl.textContent = s.level === 'high' ? 'OK'
            : s.level === 'medium' ? 'WARN'
            : s.level === 'low' ? 'RISK' : '?';

        row.appendChild(indexBadge);
        row.appendChild(textEl);
        row.appendChild(valueEl);
        row.appendChild(bar);
        row.appendChild(levelEl);
        panel.appendChild(row);
    });

    toggle.addEventListener('click', () => {
        const isOpen = !panel.hidden;
        panel.hidden = isOpen;
        toggle.textContent = isOpen ? '▸ Sentence Analysis' : '▾ Sentence Analysis';
    });

    messageEl.appendChild(toggle);
    messageEl.appendChild(panel);
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    promptInput.value = '';
    promptInput.style.height = 'auto';
    sendButton.disabled = true;
    sendButton.style.opacity = '0.5';

    addMessage(prompt, true);
    const indicator = addTypingIndicator();

    statusText.style.color = '';

    const baseUrl = getBaseUrl();

    const startTime = Date.now();
    let interval = null;
    let t1 = null;
    let t2 = null;

    try {
        if (selectedMode === 'full') {
            // ── Full Semantic Energy ──────────────────────────────────────────
            statusText.textContent = "Generating multiple diverse responses...";
            let dots = 0;
            interval = setInterval(() => {
                dots = (dots + 1) % 4;
                statusText.textContent = "Generating multiple diverse responses" + ".".repeat(dots);
            }, 500);
            t1 = setTimeout(() => { statusText.textContent = "Running Semantic Verifier Model...";      }, 7000);
            t2 = setTimeout(() => { statusText.textContent = "Calculating Fermi-Dirac Logit Flows..."; }, 14000);

            const response = await apiFetch(`${baseUrl}/chat`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ prompt, num_samples: 5, model_id: selectedModelId }),
            });
            if (!response.ok) throw new Error(`Server error ${response.status}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const msgEl = addMessageWithSentenceScores(data.answer || '(No response received)', data.sentence_scores);
            appendConfidenceBadge(msgEl, {
                score:    data.sentence_avg_confidence != null ? data.sentence_avg_confidence : data.confidence_score,
                level:    data.confidence_level,
                clusters: data.clusters_found,
                mode:     'Full SE',
            });
            appendTimer(msgEl, elapsed);
            appendMetricsPanel(msgEl, {
                type:          'full_se',
                clusters_found: data.clusters_found,
                energies:       data.debug_data?.energies_per_cluster || [],
                sentence_avg_confidence: data.sentence_avg_confidence,
            });
            appendSentenceAnalysis(msgEl, data.sentence_scores);
            recordScore(prompt, 1.0 - (data.confidence_score || 0.5), 'Full SE');

        } else if (selectedMode === 'slt') {
            // ── Fast SLT ──────────────────────────────────────────────────────
            statusText.textContent = "Generating response for SLT probe scoring...";

            const response = await apiFetch(`${baseUrl}/score_fast_slt`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ prompt }),
            });
            if (!response.ok) throw new Error(`Server error ${response.status}`);
            const data = await response.json();
            // Non-fatal: backend may return a warning (e.g. short answer) with valid fallback scores
            if (data.error && !data.answer) throw new Error(data.error);

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const msgEl = addMessageWithSentenceScores(data.answer || '(No response received)', data.sentence_scores);
            appendConfidenceBadge(msgEl, {
                score: data.sentence_avg_confidence != null ? data.sentence_avg_confidence : (1.0 - data.combined_risk),
                level: data.confidence_level,
                mode:  'Fast SLT',
            });
            appendTimer(msgEl, elapsed);
            appendMetricsPanel(msgEl, {
                type:         'probe',
                energy_risk:  data.energy_risk,
                entropy_risk: data.entropy_risk,
                combined_risk: data.combined_risk,
                sentence_avg_confidence: data.sentence_avg_confidence,
            });
            appendSentenceAnalysis(msgEl, data.sentence_scores);
            recordScore(prompt, data.combined_risk, 'Fast SLT');

        } else if (selectedMode === 'tbg') {
            // ── Fast TBG (two-phase) ──────────────────────────────────────────
            statusText.textContent = "Phase 1/2 — Pre-generation TBG probe...";

            const tbgResponse = await apiFetch(`${baseUrl}/score_fast_tbg`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ prompt }),
            });
            if (!tbgResponse.ok) throw new Error(`Server error ${tbgResponse.status}`);
            const tbgData = await tbgResponse.json();
            if (tbgData.error) throw new Error(tbgData.error);
            if (typeof tbgData.combined_risk !== 'number') throw new Error('Invalid TBG response from server');

            statusText.textContent = "Phase 2/2 — Generating response...";

            const sltResponse = await apiFetch(`${baseUrl}/score_fast_slt`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ prompt }),
            });
            if (!sltResponse.ok) throw new Error(`Server error ${sltResponse.status}`);
            const sltData = await sltResponse.json();
            // Non-fatal: backend may return a warning (e.g. short answer) with valid fallback scores
            if (sltData.error && !sltData.answer) throw new Error(sltData.error);

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const msgEl = addMessageWithSentenceScores(sltData.answer || '(No response received)', sltData.sentence_scores);
            appendConfidenceBadge(msgEl, {
                score: sltData.sentence_avg_confidence != null ? sltData.sentence_avg_confidence : (1.0 - tbgData.combined_risk),
                level: sltData.confidence_level || tbgData.confidence_level,
                mode:  'Fast TBG ⚡',
            });
            appendTimer(msgEl, elapsed);
            appendMetricsPanel(msgEl, {
                type:         'probe',
                energy_risk:  tbgData.energy_risk,
                entropy_risk: tbgData.entropy_risk,
                combined_risk: tbgData.combined_risk,
                sentence_avg_confidence: sltData.sentence_avg_confidence,
            });
            appendSentenceAnalysis(msgEl, sltData.sentence_scores);
            recordScore(prompt, tbgData.combined_risk, 'Fast TBG');
        }

        persistMessages();
        statusText.textContent = "Ready";

    } catch (error) {
        addMessage("Error: " + error.message, false);
        statusText.textContent = "Error — see message above";
        statusText.style.color = 'var(--conf-low-text)';
        setTimeout(() => { statusText.style.color = ''; statusText.textContent = "Ready"; }, 4000);
    } finally {
        // Guaranteed cleanup — runs even if the catch block throws
        if (indicator && indicator.parentNode) indicator.remove();
        if (interval) { clearInterval(interval); interval = null; }
        if (t1)       { clearTimeout(t1);        t1 = null; }
        if (t2)       { clearTimeout(t2);        t2 = null; }
        sendButton.disabled = false;
        // Restore opacity based on actual input state, not assumed state
        sendButton.style.opacity = promptInput.value.trim() ? '1' : '0.5';
        scrollToBottom();
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadPrefs();
restoreMessages();
