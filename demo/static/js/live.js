/* ============================================================
   LifeSim Demo — Live Generation Page
   Incremental event generation via OnlineLifeEventEngine
   ============================================================ */

/* ---- Theme colors (shared with trajectory.js) ---- */
const THEME = {
  childcare:     { fill: '#10b981', border: '#059669', bg: 'rgba(16,185,129,0.12)',  text: '#059669' },
  dining:        { fill: '#f59e0b', border: '#d97706', bg: 'rgba(245,158,11,0.12)',  text: '#d97706' },
  education:     { fill: '#6366f1', border: '#4f46e5', bg: 'rgba(99,102,241,0.12)',  text: '#4f46e5' },
  eldercare:     { fill: '#8b5cf6', border: '#7c3aed', bg: 'rgba(139,92,246,0.12)',  text: '#7c3aed' },
  entertainment: { fill: '#ec4899', border: '#be185d', bg: 'rgba(236,72,153,0.12)',  text: '#be185d' },
  mental_health: { fill: '#14b8a6', border: '#0d9488', bg: 'rgba(20,184,166,0.12)',  text: '#0d9488' },
  sport_health:  { fill: '#3b82f6', border: '#2563eb', bg: 'rgba(59,130,246,0.12)',  text: '#2563eb' },
  travel:        { fill: '#06b6d4', border: '#0891b2', bg: 'rgba(6,182,212,0.12)',   text: '#0891b2' },
  base:          { fill: '#94a3b8', border: '#64748b', bg: 'rgba(148,163,184,0.12)', text: '#64748b' },
};

function tc(theme) {
  return THEME[theme] || THEME.base;
}

const USER_NAMES = window.__USER_NAMES__ || {};
function userName(seqId) { return USER_NAMES[seqId] || seqId; }

/* ---- State ---- */
const S = {
  // Live sequence state
  liveSeqId:       null,
  liveProfile:     null,
  liveSequence:    null,   // {sequence_id, theme, longterm_goal}
  totalPoints:     0,
  generatedCount:  0,
  generatedNodes:  [],     // serialized nodes received from server
  generating:      false,
  done:            false,
  // Map state
  map:             null,
  markers:         [],
  polylineLatlngs: [],
  polyline:        null,
  ghostMarker:     null,
  // Selection / chat
  selectedNodeIndex: null,
  chatHistoryCache:  {},   // { nodeIndex: [{role, content, mode?, emotion?}] }
  customProfile:     null,
};

/* ---- DOM ---- */
const $ = (id) => document.getElementById(id);

const seqSelect          = $('sequence-select');
const seqMeta            = $('seq-meta');
const profileDemo        = $('profile-demo');
const generateBtn        = $('generate-btn');
const generateLabel      = $('generate-label');
const liveProgress       = $('live-progress');
const liveProgressFill   = $('live-progress-fill');
const liveProgressText   = $('live-progress-text');
const nodeOverlay        = $('node-overlay');
const ovSeq              = $('ov-seq');
const ovTheme            = $('ov-theme');
const ovNodes            = $('ov-nodes');
const nodeDrawer         = $('node-drawer');
const drawerBadge        = $('drawer-badge');
const drawerTitle        = $('drawer-title');
const drawerSub          = $('drawer-sub');
const drawerClose        = $('drawer-close');
const nodeMeta           = $('node-meta');
const eventCard          = $('event-card');
const chatFab            = $('chat-fab');
const btnOpenChat        = $('btn-open-chat');
const chatModal          = $('chat-modal');
const btnCloseChat       = $('btn-close-chat');
const chatModalTitle     = $('chat-modal-title');
const chatMessages       = $('chat-messages');
const chatInput          = $('chat-input');
const sendBtn            = $('send-btn');
const clearBtn           = $('clear-btn');
const terminalBody       = $('live-terminal-body');
const btnEditProfile     = $('btn-edit-profile');
const profileEditModal   = $('profile-edit-modal');
const btnCloseProfileEdit  = $('btn-close-profile-edit');
const btnCancelProfileEdit = $('btn-cancel-profile-edit');
const btnSaveProfile     = $('btn-save-profile');
const btnResetProfile    = $('btn-reset-profile');
const profileEditedBadge = $('profile-edited-badge');

/* ---- Init ---- */
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  bindEvents();

  window.onLangChange = () => {
    if (S.liveSequence) {
      renderSidebar(S.customProfile || S.liveProfile, S.liveSequence);
    }
    if (S.selectedNodeIndex !== null) {
      const node = S.generatedNodes[S.selectedNodeIndex];
      if (node) updateOverlayForNode(node);
    } else {
      resetOverlayLabels();
    }
  };
});

function bindEvents() {
  seqSelect.addEventListener('change', (e) => {
    if (e.target.value) startSequence(e.target.value);
  });

  generateBtn.addEventListener('click', generateNextEvent);

  drawerClose.addEventListener('click', closeDrawer);
  $('map-area').addEventListener('click', (e) => {
    if (!chatModal.classList.contains('hidden')) return;
    if (e.target === $('map') || e.target.closest('.leaflet-tile-pane')) closeDrawer();
  });

  btnOpenChat.addEventListener('click', openChatModal);
  btnCloseChat.addEventListener('click', closeChatModal);
  chatModal.addEventListener('click', (e) => { if (e.target === chatModal) closeChatModal(); });

  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') sendMessage();
  });

  btnEditProfile.addEventListener('click', openProfileEditModal);
  btnCloseProfileEdit.addEventListener('click', closeProfileEditModal);
  btnCancelProfileEdit.addEventListener('click', closeProfileEditModal);
  btnSaveProfile.addEventListener('click', saveProfile);
  btnResetProfile.addEventListener('click', resetProfile);
  profileEditModal.addEventListener('click', (e) => {
    if (e.target === profileEditModal) closeProfileEditModal();
  });

  clearBtn.addEventListener('click', async () => {
    await fetch('/api/live/reset', { method: 'POST' });
    resetLocalState();
    showToast(t('toast.session_reset'), 'info');
  });
}

/* ---- Map ---- */
function initMap() {
  S.map = L.map('map', { zoomControl: false }).setView([35.68, 139.75], 11);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(S.map);
  L.control.zoom({ position: 'bottomright' }).addTo(S.map);
}

/* ---- Start sequence ---- */
async function startSequence(seqId) {
  const res  = await fetch('/api/live/start', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ sequence_id: seqId }),
  });
  const data = await res.json();
  if (!data.success) { showToast(t('live.fail_start') + (data.error || 'unknown'), 'error'); return; }

  // Reset local state
  resetLocalState(false);  // don't call server reset again

  S.liveSeqId    = seqId;
  S.liveProfile  = data.user_profile;
  S.liveSequence = data.sequence;
  S.totalPoints  = data.total_points;

  // Render sidebar
  renderSidebar(S.customProfile || S.liveProfile, S.liveSequence);
  renderOverlay();

  // Enable generate button
  generateBtn.disabled = false;
  liveProgress.classList.remove('hidden');
  updateProgress(0, S.totalPoints);

  // Fit map to first location
  if (data.first_location && isValidCoord(data.first_location.latitude, data.first_location.longitude)) {
    S.map.setView([data.first_location.latitude, data.first_location.longitude], 13, { animate: true });
    $('map-empty').classList.add('hidden');
  }
}

/* ---- Generate next event ---- */
async function generateNextEvent() {
  if (S.generating || S.done || !S.liveSeqId) return;

  S.generating = true;
  generateBtn.disabled = true;
  terminalClear();

  // Show spinner immediately
  generateLabel.textContent = t('live.generating');
  const spinIcon = $('generate-icon');
  if (spinIcon) {
    spinIcon.innerHTML = `
      <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2.5"
        stroke-dasharray="28" stroke-dashoffset="10"
        fill="none" style="transform-origin:50% 50%;animation:spinIcon 0.8s linear infinite"/>`;
    if (!document.getElementById('spin-keyframes')) {
      const style = document.createElement('style');
      style.id = 'spin-keyframes';
      style.textContent = '@keyframes spinIcon{to{transform:rotate(360deg)}}';
      document.head.appendChild(style);
    }
  }
  terminalLine(t('live.calling_llm'), 'tl-step');

  // Use fetch + readable stream to receive SSE in real time
  const res = await fetch('/api/live/generate-event', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({}),
  });

  let data = null;
  try {
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Parse complete SSE events (separated by double newline)
      const parts = buf.split('\n\n');
      buf = parts.pop();  // keep any incomplete trailing chunk

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data: ')) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.type === 'log') {
            terminalLine(msg.line, classForLine(msg.line));
          } else if (msg.type === 'result') {
            data = msg;
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    terminalLine('[error] stream read failed: ' + err.message, 'tl-warn');
  }

  // Restore button / icon
  S.generating = false;
  generateLabel.textContent = t('live.gen_btn');
  const icon = $('generate-icon');
  if (icon) icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';

  if (!data) {
    terminalLine(t('live.no_result_log'), 'tl-warn');
    showToast(t('live.fail_no_result'), 'error');
    generateBtn.disabled = false;
    return;
  }

  if (!data.success) {
    terminalLine('✗ Failed.', 'tl-warn');
    showToast(t('live.fail_gen') + (data.error || 'unknown'), 'error');
    generateBtn.disabled = false;
    return;
  }

  terminalLine(t('live.done_log'), 'tl-ok');

  const node = data.node;
  S.generatedNodes.push(node);
  S.generatedCount = data.progress.current;
  S.done = data.done;

  addNodeToMap(node);
  updateProgress(data.progress.current, data.progress.total);
  renderOverlay();
  selectGeneratedNode(node.node_index);

  if (S.done) {
    generateBtn.disabled = true;
    generateLabel.textContent = t('live.all_done_btn');
    showToast(t('live.all_done_toast'), 'info');
  } else {
    generateBtn.disabled = false;
  }
}

function classForLine(line) {
  if (!line) return 'tl-info';
  const l = line.toLowerCase();
  if (l.startsWith('[error]'))                   return 'tl-warn';
  if (l.startsWith('[warn]'))                    return 'tl-warn';
  if (l.includes('selected event'))              return 'tl-ok';
  if (l.includes('generated query'))             return 'tl-prompt';
  if (l.includes('total candidate'))             return 'tl-step';
  if (l.includes('rerank'))                      return 'tl-step';
  if (l.includes('retriev'))                     return 'tl-step';
  return 'tl-info';
}

/* ---- Map rendering (incremental) ---- */
function addNodeToMap(node) {
  const loc = node.location_detail;
  if (!loc || !isValidCoord(loc.latitude, loc.longitude)) return;

  $('map-empty').classList.add('hidden');
  const theme = S.liveSequence ? S.liveSequence.theme : 'base';
  const color = tc(theme);
  const latlng = [loc.latitude, loc.longitude];

  S.polylineLatlngs.push(latlng);

  // Rebuild polyline
  if (S.polyline) S.map.removeLayer(S.polyline);
  if (S.polylineLatlngs.length >= 2) {
    S.polyline = L.polyline(S.polylineLatlngs, {
      color:     color.fill,
      weight:    3,
      opacity:   0.5,
      dashArray: '8 6',
      lineCap:   'round',
    }).addTo(S.map);
  }

  // Add marker
  const marker = L.marker(latlng, {
    icon: makeIcon(node.node_index, false, color),
  })
    .bindTooltip(
      `<b>N${node.node_index + 1}</b> · ${node.location || ''}<br>${truncate(node.event || node.life_event || '', 60)}`,
      { className: 'ls-tooltip', sticky: true }
    )
    .addTo(S.map);

  marker.nodeIndex = node.node_index;
  marker.on('click', () => selectGeneratedNode(node.node_index));
  S.markers.push(marker);

  // Pan to new node
  S.map.setView(latlng, Math.max(S.map.getZoom(), 13), { animate: true });
}

function makeIcon(nodeIndex, active, color) {
  const size = active ? 36 : 28;
  const bg   = active ? color.fill : '#ffffff';
  const txt  = active ? '#ffffff' : color.fill;
  const ring = active
    ? `0 0 0 3px ${color.fill}55, 0 4px 14px rgba(0,0,0,0.25)`
    : '0 2px 8px rgba(0,0,0,0.18)';
  return L.divIcon({
    className: '',
    html: `<div class="map-marker" style="
      width:${size}px; height:${size}px;
      background:${bg}; color:${txt};
      border:2.5px solid ${color.fill};
      box-shadow:${ring};
      font-size:${active ? 13 : 11}px;
    ">${nodeIndex + 1}</div>`,
    iconSize:   [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function refreshAllMarkers() {
  const theme = S.liveSequence ? S.liveSequence.theme : 'base';
  const color = tc(theme);
  S.markers.forEach(m => {
    const active = m.nodeIndex === S.selectedNodeIndex;
    m.setIcon(makeIcon(m.nodeIndex, active, color));
  });
}

/* ---- Ghost marker ---- */
function updateGhostMarker(locationInfo) {
  if (!isValidCoord(locationInfo.latitude, locationInfo.longitude)) return;
  removeGhostMarker();

  const ghostIcon = L.divIcon({
    className: '',
    html: `<div class="ghost-marker">
      <span class="ghost-marker-label">${esc(locationInfo.location || 'Next')}</span>
      <div class="ghost-marker-pulse"></div>
    </div>`,
    iconSize:   [32, 32],
    iconAnchor: [16, 16],
  });

  S.ghostMarker = L.marker(
    [locationInfo.latitude, locationInfo.longitude],
    { icon: ghostIcon, interactive: false, zIndexOffset: -100 }
  ).addTo(S.map);
}

function removeGhostMarker() {
  if (S.ghostMarker) {
    S.map.removeLayer(S.ghostMarker);
    S.ghostMarker = null;
  }
}

/* ---- Overlay ---- */
function renderOverlay() {
  ovSeq.textContent   = S.liveSeqId ? userName(S.liveSeqId) : '—';
  ovTheme.textContent = S.liveSequence ? S.liveSequence.theme : '—';
  ovNodes.textContent = S.generatedNodes.length
    ? `${S.generatedNodes.length} / ${S.totalPoints}`
    : '—';
}

function updateOverlayForNode(node) {
  ovSeq.textContent   = `N${node.node_index + 1}`;
  ovTheme.textContent = node.time || '—';
  ovNodes.textContent = node.weather || '—';
  const rows = nodeOverlay.querySelectorAll('.ov-label');
  if (rows[0]) rows[0].textContent = t('ov.node');
  if (rows[1]) rows[1].textContent = t('ov.time');
  if (rows[2]) rows[2].textContent = t('ov.weather');
}

function resetOverlayLabels() {
  const rows = nodeOverlay.querySelectorAll('.ov-label');
  if (rows[0]) rows[0].textContent = t('ov.user');
  if (rows[1]) rows[1].textContent = t('ov.theme');
  if (rows[2]) rows[2].textContent = t('live.events_lbl');
  renderOverlay();
}

/* ---- Progress bar ---- */
function updateProgress(current, total) {
  const pct = total > 0 ? (current / total) * 100 : 0;
  liveProgressFill.style.width = `${pct}%`;
  liveProgressText.textContent = `${current} / ${total}${t('live.progress_unit')}`;
}

/* ---- Sidebar ---- */
function renderSidebar(profile, sequence) {
  const p = profile || {};
  const color = tc(sequence ? sequence.theme : 'base');

  seqMeta.innerHTML = `
    <div class="seq-meta-row"><span class="seq-meta-label">${t('sb.theme')}</span>
      <span class="seq-meta-value">
        <span class="tag tag-purple" style="background:${color.bg};color:${color.text};">${esc(sequence ? sequence.theme : t('misc.na'))}</span>
      </span>
    </div>
    ${sequence && sequence.longterm_goal ? `
    <div class="seq-meta-row"><span class="seq-meta-label" style="width:auto;margin-right:6px;">${t('sb.goal')}</span>
      <span class="seq-meta-value" style="font-size:11px;">${esc(truncate(sequence.longterm_goal, 60))}</span>
    </div>` : ''}
  `;

  const personality = (p.personality || []).slice(0, 8);
  const prefs       = (p.preferences || []).slice(0, 5);

  profileDemo.innerHTML = `
    ${prow(t('pe.gender'), p.gender)}
    ${prow(t('pe.age'), p.age)}
    ${prow(t('pe.area'), p.area)}
    ${prow(t('pe.employment'), p.employment)}
    ${prow(t('pe.income'), p.income)}
    ${prow(t('pe.marital'), p.marital)}
    ${personality.length ? `
      <div style="margin-top:12px;">
        <div class="sb-section-title" style="font-size:10px;color:var(--text-muted);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:7px;">${t('sb.personality')}</div>
        <div class="tag-wrap">${personality.map(tr => `<span class="tag tag-purple">${esc(tr)}</span>`).join('')}</div>
      </div>` : ''}
    ${prefs.length ? `
      <div style="margin-top:12px;">
        <div class="sb-section-title" style="font-size:10px;color:var(--text-muted);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:7px;">${t('sb.prefs')}</div>
        <div class="tag-wrap">${prefs.map(tr => `<span class="tag tag-green">${esc(truncate(tr, 50))}</span>`).join('')}</div>
      </div>` : ''}
  `;
}

function prow(label, value) {
  if (!value) return '';
  return `<div class="profile-row">
    <span class="profile-label">${label}</span>
    <span class="profile-value">${esc(String(value))}</span>
  </div>`;
}

/* ---- Select generated node ---- */
async function selectGeneratedNode(nodeIndex) {
  const res  = await fetch('/api/live/select-node', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ node_index: nodeIndex }),
  });
  const data = await res.json();
  if (!data.success) { showToast(t('live.fail_node'), 'error'); return; }

  S.selectedNodeIndex = nodeIndex;

  if (!S.chatHistoryCache[nodeIndex]) {
    S.chatHistoryCache[nodeIndex] = data.chat_history || [];
  }

  closeChatModal();
  refreshAllMarkers();
  updateOverlayForNode(data.selected_node);
  openDrawer(data.selected_node);
}

/* ---- Drawer ---- */
function openDrawer(node) {
  const theme = S.liveSequence ? S.liveSequence.theme : 'base';
  const color = tc(theme);

  drawerBadge.textContent      = node.node_index + 1;
  drawerBadge.style.background = color.fill;
  drawerTitle.textContent      = `${t('ov.node')} ${node.node_index + 1} · ${esc(theme)}`;
  drawerSub.textContent        = [node.time, node.location].filter(Boolean).join(' · ');
  nodeDrawer.classList.add('open');
  chatFab.classList.remove('hidden');

  nodeMeta.innerHTML = `
    ${mrow(t('ov.time'), node.time)}
    ${mrow(t('dr.location'), node.location)}
    ${mrow(t('ov.weather'), node.weather)}
    ${mrow(t('dr.intent'), node.intent)}
  `;

  eventCard.textContent = node.event || node.life_event || 'Generating…';
  eventCard.style.borderLeftColor = color.fill;
}

function closeDrawer() {
  closeChatModal();
  chatFab.classList.add('hidden');
  nodeDrawer.classList.remove('open');
  if (S.selectedNodeIndex !== null) {
    S.selectedNodeIndex = null;
    refreshAllMarkers();
    resetOverlayLabels();
  }
}

function mrow(label, value) {
  if (!value) return '';
  return `<span class="node-meta-label">${label}</span><span class="node-meta-value">${esc(value)}</span>`;
}


/* ---- Chat modal ---- */
function openChatModal() {
  if (S.selectedNodeIndex === null) return;
  const node = S.generatedNodes[S.selectedNodeIndex];
  chatModalTitle.textContent = node
    ? `N${node.node_index + 1} · ${node.location || ''}`
    : t('chat.title');

  const history = S.chatHistoryCache[S.selectedNodeIndex] || [];
  chatModal.classList.remove('hidden');
  renderChatHistory(history);

  const ended = history.length > 0 && history[history.length - 1]._ended;
  chatInput.disabled = !!ended;
  sendBtn.disabled   = !!ended;
  if (!ended) chatInput.focus();
}

function closeChatModal() {
  chatModal.classList.add('hidden');
}

function renderChatHistory(history) {
  chatMessages.innerHTML = '';
  if (!history || !history.length) {
    chatMessages.innerHTML = `<div class="chat-empty-state">${t('chat.empty')}</div>`;
    return;
  }

  history.forEach(item => {
    if (item._ended) return;
    appendBubble(item.role, item.content, item.mode || null, item.emotion || null);
  });
}

function appendBubble(role, content, mode = null, emotion = null) {
  const empty = chatMessages.querySelector('.chat-empty-state');
  if (empty) empty.remove();

  const isUser = role === 'user';
  const wrap   = document.createElement('div');
  wrap.className = `chat-msg ${isUser ? 'chat-msg-user' : 'chat-msg-assist'} fade-up`;

  const avatar = document.createElement('div');
  avatar.className = 'chat-avatar';
  avatar.textContent = isUser ? t('chat.you') : t('chat.u');

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.innerHTML = renderMarkdown(content);

  if (mode === 'fallback') {
    bubble.innerHTML += `<div class="chat-fallback-badge">${t('chat.fallback')}</div>`;
  }
  if (!isUser && emotion && emotion !== 'neutral') {
    const tag = document.createElement('div');
    tag.className = 'emotion-tag';
    tag.textContent = emotion;
    bubble.appendChild(tag);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function markConversationEnded() {
  chatInput.disabled = true;
  sendBtn.disabled   = true;
  const divider = document.createElement('div');
  divider.className = 'chat-ended-notice';
  divider.textContent = t('chat.ended');
  chatMessages.appendChild(divider);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg || S.selectedNodeIndex === null) return;

  const idx   = S.selectedNodeIndex;
  const cache = S.chatHistoryCache[idx] = S.chatHistoryCache[idx] || [];

  chatInput.value    = '';
  chatInput.disabled = true;
  sendBtn.disabled   = true;

  appendBubble('user', msg);
  cache.push({ role: 'user', content: msg });

  const typingWrap = document.createElement('div');
  typingWrap.className = 'chat-msg chat-msg-assist';
  typingWrap.innerHTML = `
    <div class="chat-avatar">${t('chat.u')}</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  chatMessages.appendChild(typingWrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const res = await fetch('/api/live/chat', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ node_index: idx, message: msg }),
  });

  const data = await res.json();

  if (data.success) {
    const ended = data.action === 'End Conversation';
    typingWrap.remove();
    if (data.response) {
      appendBubble('assistant', data.response, data.mode, data.emotion || null);
      cache.push({ role: 'assistant', content: data.response, mode: data.mode, emotion: data.emotion });
    }
    if (ended) {
      cache.push({ _ended: true });
      markConversationEnded();
      return;
    }
  } else {
    typingWrap.remove();
    cache.pop();
    showToast(t('toast.send_fail') + (data.error || 'unknown'), 'error');
  }

  chatInput.disabled = false;
  sendBtn.disabled   = false;
  chatInput.focus();
}

/* ---- Profile editing ---- */
function effectiveProfile() {
  return S.customProfile || S.liveProfile || {};
}

function updateProfileEditedBadge() {
  profileEditedBadge.classList.toggle('hidden', !S.customProfile);
}

function openProfileEditModal() {
  const p = effectiveProfile();
  document.getElementById('pe-gender').value     = p.gender     || '';
  document.getElementById('pe-age').value        = p.age        || '';
  document.getElementById('pe-area').value       = p.area       || '';
  document.getElementById('pe-employment').value = p.employment || '';
  document.getElementById('pe-income').value     = p.income     || '';
  document.getElementById('pe-marital').value    = p.marital    || '';
  document.getElementById('pe-personality').value = (p.personality || []).join('\n');
  document.getElementById('pe-preferences').value = (p.preferences || []).join('\n');
  profileEditModal.classList.remove('hidden');
}

function closeProfileEditModal() {
  profileEditModal.classList.add('hidden');
}

async function saveProfile() {
  const base  = effectiveProfile();
  const lines = (id) => document.getElementById(id).value
    .split('\n').map(s => s.trim()).filter(Boolean);

  const updated = {
    ...base,
    gender:      document.getElementById('pe-gender').value.trim(),
    age:         document.getElementById('pe-age').value.trim(),
    area:        document.getElementById('pe-area').value.trim(),
    employment:  document.getElementById('pe-employment').value.trim(),
    income:      document.getElementById('pe-income').value.trim(),
    marital:     document.getElementById('pe-marital').value.trim(),
    personality: lines('pe-personality'),
    preferences: lines('pe-preferences'),
  };

  const res = await fetch('/api/update-profile', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ profile: updated }),
  });
  const data = await res.json();
  if (!data.success) { showToast(t('toast.profile_save_fail'), 'error'); return; }

  S.customProfile    = updated;
  S.chatHistoryCache = {};
  updateProfileEditedBadge();
  closeProfileEditModal();
  if (S.liveSequence) renderSidebar(updated, S.liveSequence);
  showToast(t('toast.profile_saved'), 'info');
}

async function resetProfile() {
  const res = await fetch('/api/update-profile', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ profile: null }),
  });
  const data = await res.json();
  if (!data.success) { showToast(t('toast.profile_reset_fail'), 'error'); return; }

  S.customProfile    = null;
  S.chatHistoryCache = {};
  updateProfileEditedBadge();
  closeProfileEditModal();
  if (S.liveSequence) renderSidebar(S.liveProfile, S.liveSequence);
  showToast(t('toast.profile_reset'), 'info');
}

/* ---- Local state reset ---- */
function resetLocalState(callServerReset = false) {
  // Clear map
  S.markers.forEach(m => S.map.removeLayer(m));
  S.markers = [];
  if (S.polyline) { S.map.removeLayer(S.polyline); S.polyline = null; }
  S.polylineLatlngs = [];
  removeGhostMarker();

  // Clear state
  S.liveSeqId        = null;
  S.liveProfile      = null;
  S.liveSequence     = null;
  S.totalPoints      = 0;
  S.generatedCount   = 0;
  S.generatedNodes   = [];
  S.generating       = false;
  S.done             = false;
  S.selectedNodeIndex = null;
  S.chatHistoryCache = {};
  S.customProfile    = null;

  // Reset UI
  terminalClear();
  terminalBody.innerHTML = `<span class="live-terminal-empty">${t('live.waiting')}</span>`;
  seqSelect.value = '';
  seqMeta.innerHTML = '';
  profileDemo.textContent = t('live.select_begin');
  generateBtn.disabled = true;
  generateLabel.textContent = t('live.gen_btn');
  liveProgress.classList.add('hidden');
  updateProfileEditedBadge();
  closeDrawer();
  closeChatModal();
  $('map-empty').classList.remove('hidden');
  resetOverlayLabels();
}

/* ---- Terminal log window ---- */
function terminalClear() {
  terminalBody.innerHTML = '';
}

function terminalLine(text, cls = 'tl-info') {
  const empty = terminalBody.querySelector('.live-terminal-empty');
  if (empty) empty.remove();
  const span = document.createElement('span');
  span.className = `tl ${cls}`;
  span.textContent = text;
  terminalBody.appendChild(span);
  terminalBody.scrollTop = terminalBody.scrollHeight;
}


/* ---- Countdown spinner ---- */
/* ---- Helpers ---- */
function isValidCoord(lat, lng) {
  return Number.isFinite(lat) && Number.isFinite(lng);
}

function esc(text) {
  const d = document.createElement('div');
  d.textContent = text || '';
  return d.innerHTML;
}

function truncate(text, len) {
  if (!text || text.length <= len) return text || '';
  return text.slice(0, len) + '…';
}

function renderMarkdown(text) {
  if (typeof marked !== 'undefined') return marked.parse(text || '');
  return esc(text);
}
