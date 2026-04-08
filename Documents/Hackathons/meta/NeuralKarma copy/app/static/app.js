/**
 * NeuralKarma — Frontend Application
 * Real-time ethical impact scoring dashboard.
 * All data fetched from trained ML models via REST API — no mock values.
 */

// ─── Configuration ──────────────────────────────────────
const API_BASE = '';
const WS_URL = `ws://${window.location.host}/ws/live`;

// ─── State ──────────────────────────────────────────────
let ws = null;
let radarChart = null;
let currentUsername = 'anonymous';
let normsPage = 1;
let normsTotalPages = 1;

// ─── Initialize ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadStats();
    loadLeaderboard();
    loadNorms();
    loadDatasetInfo();
    setupEventListeners();
});

// ─── WebSocket ──────────────────────────────────────────
function initWebSocket() {
    const indicator = document.getElementById('ws-indicator');
    const statusText = document.getElementById('ws-status');

    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            indicator.classList.add('connected');
            statusText.textContent = 'Live';
            // Keep-alive ping every 30s
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'karma_update') {
                handleLiveUpdate(msg.data);
            }
        };

        ws.onclose = () => {
            indicator.classList.remove('connected');
            statusText.textContent = 'Disconnected';
            // Reconnect after 3s
            setTimeout(initWebSocket, 3000);
        };

        ws.onerror = () => {
            indicator.classList.remove('connected');
            statusText.textContent = 'Error';
        };
    } catch (e) {
        console.warn('WebSocket not available:', e);
        if (statusText) statusText.textContent = 'Offline';
    }
}

function handleLiveUpdate(data) {
    // Update stats on every live update
    loadStats();
    loadLeaderboard();
    showToast(`New karma score: ${data.aggregate_karma.toFixed(1)} (${data.tier.label})`, 'info');
}

// ─── Event Listeners ────────────────────────────────────
function setupEventListeners() {
    // Score form
    document.getElementById('score-btn').addEventListener('click', scoreAction);
    document.getElementById('action-text').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            scoreAction();
        }
    });

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const group = tab.closest('.tabs');
            const contentContainer = group.nextElementSibling || group.parentElement;
            const target = tab.dataset.tab;

            // Update active tab
            group.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update content
            const parent = group.parentElement;
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const content = parent.querySelector(`[data-tab-content="${target}"]`);
            if (content) content.classList.add('active');
        });
    });

    // Norms search
    const normsSearch = document.getElementById('norms-search');
    if (normsSearch) {
        let debounce;
        normsSearch.addEventListener('input', () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => {
                normsPage = 1;
                loadNorms();
            }, 300);
        });
    }

    // History search
    const histBtn = document.getElementById('load-history-btn');
    if (histBtn) {
        histBtn.addEventListener('click', loadHistory);
    }
}

// ─── Score Action ───────────────────────────────────────
async function scoreAction() {
    const textInput = document.getElementById('action-text');
    const usernameInput = document.getElementById('username-input');
    const btn = document.getElementById('score-btn');
    const text = textInput.value.trim();

    if (!text) {
        showToast('Please enter an action to score', 'error');
        return;
    }

    const username = usernameInput.value.trim() || 'anonymous';
    currentUsername = username;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scoring...';

    try {
        const res = await fetch(`${API_BASE}/api/score`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, username }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        displayResult(data);
        showToast(`Karma scored: ${data.aggregate_karma.toFixed(1)} — ${data.tier.label}`, 'success');

        // Refresh dependent panels
        loadStats();
        loadLeaderboard();
        loadHistory();

    } catch (err) {
        showToast(`Scoring failed: ${err.message}`, 'error');
        console.error('Score error:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = ' Score Karma';
    }
}

// ─── Display Result ─────────────────────────────────────
function displayResult(data) {
    const container = document.getElementById('result-container');
    container.classList.add('active');

    // Karma Score Circle
    const scoreValue = document.getElementById('karma-score-value');
    animateCounter(scoreValue, data.aggregate_karma, 1);

    // Tier Badge
    const tierBadge = document.getElementById('tier-badge');
    tierBadge.style.background = `${data.tier.color}20`;
    tierBadge.style.color = data.tier.color;
    tierBadge.style.border = `1px solid ${data.tier.color}40`;
    tierBadge.textContent = `${data.tier.tier} — ${data.tier.label}`;

    // Tier Description
    document.getElementById('tier-description').textContent = data.tier.description;

    // Confidence
    const confFill = document.getElementById('confidence-fill');
    confFill.style.width = `${(data.confidence * 100).toFixed(0)}%`;
    document.getElementById('confidence-text').textContent =
        `${(data.confidence * 100).toFixed(1)}% confidence`;

    // Update radar chart
    updateRadarChart(data.axis_scores);

    // Update axis list
    updateAxisList(data.axis_scores);

    // Update ripple visualization
    updateRipple(data.ripple);
}

// ─── Radar Chart (Chart.js) ─────────────────────────────
function updateRadarChart(axisScores) {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;

    const labels = Object.keys(axisScores).map(k =>
        k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    );
    const values = Object.values(axisScores);

    if (radarChart) {
        radarChart.data.labels = labels;
        radarChart.data.datasets[0].data = values;
        radarChart.update('none');
        // Animate update
        setTimeout(() => radarChart.update(), 50);
        return;
    }

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Karma Axes',
                data: values,
                backgroundColor: 'rgba(108, 92, 231, 0.12)',
                borderColor: 'rgba(108, 92, 231, 0.7)',
                pointBackgroundColor: '#6C5CE7',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7,
                borderWidth: 2,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        color: 'rgba(0, 0, 0, 0.3)',
                        backdropColor: 'transparent',
                        font: { size: 10 },
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.06)',
                    },
                    angleLines: {
                        color: 'rgba(0, 0, 0, 0.06)',
                    },
                    pointLabels: {
                        color: 'rgba(0, 0, 0, 0.6)',
                        font: { size: 11, weight: '600' },
                    },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: 'rgba(108, 92, 231, 0.2)',
                    borderWidth: 1,
                    titleColor: '#2D2B29',
                    bodyColor: '#5C5955',
                    padding: 12,
                    cornerRadius: 8,
                },
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart',
            },
        },
    });
}

function updateAxisList(axisScores) {
    const list = document.getElementById('axis-list');
    if (!list) return;

    list.innerHTML = '';
    const axisColors = {
        prosociality: '#6366f1',
        harm_avoidance: '#10b981',
        fairness: '#f59e0b',
        virtue: '#ec4899',
        duty: '#06b6d4',
    };

    for (const [axis, score] of Object.entries(axisScores)) {
        const color = axisColors[axis] || '#6366f1';
        const scoreClass = score >= 70 ? 'score-good' :
                          score >= 45 ? 'score-neutral' :
                          score >= 25 ? 'score-warning' : 'score-danger';

        list.innerHTML += `
            <div class="axis-item">
                <span class="axis-name">${axis.replace(/_/g, ' ')}</span>
                <div class="axis-bar">
                    <div class="axis-bar-fill" style="width: ${score}%; background: ${color}"></div>
                </div>
                <span class="axis-score ${scoreClass}">${score.toFixed(1)}</span>
            </div>
        `;
    }
}

// ─── Ripple Effect ──────────────────────────────────────
function updateRipple(ripple) {
    if (!ripple) return;

    document.getElementById('ripple-impact').textContent =
        ripple.total_ripple_impact.toFixed(1);
    document.getElementById('ripple-people').textContent =
        ripple.total_people_reached;
    document.getElementById('ripple-damping').textContent =
        (ripple.damping_factor * 100).toFixed(0) + '%';

    // Update hop details
    const hopsContainer = document.getElementById('ripple-hops');
    if (hopsContainer && ripple.hops) {
        hopsContainer.innerHTML = ripple.hops.map(hop => `
            <div class="ripple-stat">
                <div class="ripple-stat-value">${hop.people_affected}</div>
                <div class="ripple-stat-label">Depth ${hop.depth} · ${hop.impact_per_person.toFixed(1)}/person</div>
            </div>
        `).join('');
    }
}

// ─── Stats ──────────────────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const data = await res.json();

        setElementText('stat-total-actions', data.total_actions);
        setElementText('stat-total-users', data.total_users);
        setElementText('stat-avg-karma', data.avg_karma.toFixed(1));
        setElementText('stat-ripple-impact', data.total_ripple_impact.toFixed(0));

        // Tier distribution
        updateTierDistribution(data.tier_distribution, data.total_actions);

        // Axis averages
        updateAxisAverages(data.axis_averages);

    } catch (err) {
        console.warn('Stats load failed:', err);
    }
}

function updateTierDistribution(dist, total) {
    const bar = document.getElementById('tier-bar');
    if (!bar || total === 0) return;

    const tierColors = {
        S: '#FFD700', A: '#00E5FF', B: '#69F0AE',
        C: '#B0BEC5', D: '#FFB74D', E: '#FF5252', F: '#D50000',
    };

    bar.innerHTML = '';
    for (const [tier, count] of Object.entries(dist)) {
        if (count === 0) continue;
        const pct = ((count / total) * 100).toFixed(1);
        bar.innerHTML += `
            <div class="tier-segment" style="flex: ${count}; background: ${tierColors[tier]}"
                 title="${tier}: ${count} (${pct}%)">
                ${pct > 5 ? tier : ''}
            </div>
        `;
    }
}

function updateAxisAverages(avgs) {
    const container = document.getElementById('axis-averages');
    if (!container) return;

    const colors = {
        prosociality: '#6366f1', harm_avoidance: '#10b981',
        fairness: '#f59e0b', virtue: '#ec4899', duty: '#06b6d4',
    };

    container.innerHTML = '';
    for (const [axis, avg] of Object.entries(avgs)) {
        container.innerHTML += `
            <div class="axis-item">
                <span class="axis-name">${axis.replace(/_/g, ' ')}</span>
                <div class="axis-bar">
                    <div class="axis-bar-fill" style="width: ${avg}%; background: ${colors[axis] || '#6366f1'}"></div>
                </div>
                <span class="axis-score">${avg.toFixed(1)}</span>
            </div>
        `;
    }
}

// ─── Leaderboard ────────────────────────────────────────
async function loadLeaderboard() {
    try {
        const res = await fetch(`${API_BASE}/api/leaderboard?limit=10`);
        const data = await res.json();
        const container = document.getElementById('leaderboard-list');
        if (!container) return;

        if (data.leaderboard.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    
                    <p>No users yet. Score your first action!</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.leaderboard.map(user => {
            const rankClass = user.rank === 1 ? 'gold' :
                             user.rank === 2 ? 'silver' :
                             user.rank === 3 ? 'bronze' : 'normal';
            return `
                <div class="leaderboard-row">
                    <div class="leaderboard-rank ${rankClass}">${user.rank}</div>
                    <div class="leaderboard-name">${escapeHtml(user.display_name || user.username)}</div>
                    <div class="leaderboard-actions">${user.total_actions} actions</div>
                    <div class="karma-tier-badge" style="background:${user.tier.color}20;color:${user.tier.color};border:1px solid ${user.tier.color}40;padding:2px 8px;border-radius:100px;font-size:0.75rem;">
                        ${user.tier.tier}
                    </div>
                    <div class="leaderboard-score" style="color:${user.tier.color}">
                        ${user.aggregate_karma.toFixed(1)}
                    </div>
                </div>
            `;
        }).join('');

    } catch (err) {
        console.warn('Leaderboard load failed:', err);
    }
}

// ─── History ────────────────────────────────────────────
async function loadHistory() {
    const username = document.getElementById('username-input')?.value?.trim() || currentUsername;
    if (!username || username === 'anonymous') return;

    try {
        const res = await fetch(`${API_BASE}/api/history/${encodeURIComponent(username)}?limit=20`);
        if (!res.ok) {
            if (res.status === 404) return;
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();

        const container = document.getElementById('history-timeline');
        if (!container) return;

        if (data.actions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    
                    <p>No history yet for ${escapeHtml(username)}</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.actions.map((action, i) => `
            <div class="timeline-item" style="animation-delay: ${i * 0.05}s">
                <div class="timeline-text">${escapeHtml(action.text.substring(0, 120))}${action.text.length > 120 ? '...' : ''}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
                    <span class="timeline-score" style="color:${getScoreColor(action.aggregate_score)}">
                        ${action.aggregate_score.toFixed(1)} → ${action.decayed_score.toFixed(1)} (decayed)
                    </span>
                    <span class="timeline-time">${formatTime(action.created_at)}</span>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.warn('History load failed:', err);
    }
}

// ─── Social Norms ───────────────────────────────────────
async function loadNorms() {
    const search = document.getElementById('norms-search')?.value?.trim() || '';

    try {
        const params = new URLSearchParams({
            page: normsPage,
            per_page: 15,
        });
        if (search) params.set('search', search);

        const res = await fetch(`${API_BASE}/api/norms?${params}`);
        const data = await res.json();
        normsTotalPages = data.total_pages;

        const container = document.getElementById('norms-list');
        if (!container) return;

        if (data.norms.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    
                    <p>No social norms found${search ? ` matching "${escapeHtml(search)}"` : ''}</p>
                </div>
            `;
            updateNormsPagination();
            return;
        }

        container.innerHTML = data.norms.map(norm => `
            <div class="norm-item">
                <div class="norm-text">"${escapeHtml(norm.norm)}"</div>
                <span class="norm-label ${norm.safety_label}">${norm.safety_label}</span>
                <span style="font-size:0.7rem;color:var(--text-muted);margin-left:8px">Score: ${norm.prosocial_score}</span>
            </div>
        `).join('');

        updateNormsPagination();
        setElementText('norms-total', `${data.total.toLocaleString()} norms`);

    } catch (err) {
        console.warn('Norms load failed:', err);
    }
}

function updateNormsPagination() {
    setElementText('norms-page-info', `Page ${normsPage} of ${normsTotalPages}`);
    const prevBtn = document.getElementById('norms-prev');
    const nextBtn = document.getElementById('norms-next');
    if (prevBtn) prevBtn.disabled = normsPage <= 1;
    if (nextBtn) nextBtn.disabled = normsPage >= normsTotalPages;
}

function normsPageChange(delta) {
    normsPage = Math.max(1, Math.min(normsTotalPages, normsPage + delta));
    loadNorms();
}

// ─── Dataset Info ───────────────────────────────────────
async function loadDatasetInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/dataset-info`);
        const data = await res.json();
        const container = document.getElementById('dataset-info');
        if (!container) return;

        container.innerHTML = data.datasets.map(ds => `
            <div class="dataset-card">
                <div class="dataset-name">${ds.name}</div>
                <div class="dataset-source">${ds.source}</div>
                <div class="dataset-desc">${ds.description}</div>
                ${ds.rows ? `<div style="margin-top:8px"><span class="dataset-rows">${typeof ds.rows === 'number' ? ds.rows.toLocaleString() : ds.rows}</span> rows</div>` : ''}
                ${ds.subsets ? `<div style="margin-top:8px;font-size:0.8rem;color:#8b8ba0">${Object.entries(ds.subsets).map(([k,v]) => `${k}: <b>${typeof v === 'number' ? v.toLocaleString() : v}</b>`).join(' · ')}</div>` : ''}
                <div style="margin-top:8px"><a href="${ds.paper}" target="_blank" style="font-size:0.8rem">📄 Paper</a></div>
            </div>
        `).join('');

        if (data.social_norms_count) {
            container.innerHTML += `
                <div class="dataset-card">
                    <div class="dataset-name">Extracted Social Norms</div>
                    <div class="dataset-source">Derived from ProsocialDialog</div>
                    <div class="dataset-desc">Rules-of-thumb (RoTs) — crowdworker-annotated moral norms</div>
                    <div style="margin-top:8px"><span class="dataset-rows">${typeof data.social_norms_count === 'number' ? data.social_norms_count.toLocaleString() : data.social_norms_count}</span> unique norms</div>
                </div>
            `;
        }

    } catch (err) {
        console.warn('Dataset info load failed:', err);
    }
}

// ─── Utilities ──────────────────────────────────────────
function animateCounter(element, target, decimals = 0) {
    if (!element) return;
    const start = parseFloat(element.textContent) || 0;
    const duration = 800;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;
        element.textContent = current.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function setElementText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function getScoreColor(score) {
    if (score >= 75) return '#00B894';
    if (score >= 60) return '#0984E3';
    if (score >= 45) return '#636E72';
    if (score >= 30) return '#E17055';
    return '#D63031';
}

function formatTime(isoString) {
    if (!isoString) return '';
    const d = new Date(isoString);
    const now = new Date();
    const diff = (now - d) / 1000;

    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return d.toLocaleDateString();
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s var(--ease-out) forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
