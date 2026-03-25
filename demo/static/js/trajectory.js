/* ============================================================
   LifeSim Demo — Trajectory Page
   Map-first layout: sidebar + full-screen map + slide-in drawer
   ============================================================ */

/* ---- Theme colors ---- */
const THEME = {
  childcare:     { fill: '#10b981', border: '#059669', bg: 'rgba(16,185,129,0.12)',  text: '#059669' },
  dining:        { fill: '#f59e0b', border: '#d97706', bg: 'rgba(245,158,11,0.12)',  text: '#d97706' },
  education:     { fill: '#6366f1', border: '#4f46e5', bg: 'rgba(99,102,241,0.12)',  text: '#4f46e5' },
  eldercare:     { fill: '#8b5cf6', border: '#7c3aed', bg: 'rgba(139,92,246,0.12)',  text: '#7c3aed' },
  entertainment: { fill: '#ec4899', border: '#be185d', bg: 'rgba(236,72,153,0.12)',  text: '#be185d' },
  mental_health: { fill: '#14b8a6', border: '#0d9488', bg: 'rgba(20,184,166,0.12)',  text: '#0d9488' },
  sport:         { fill: '#3b82f6', border: '#2563eb', bg: 'rgba(59,130,246,0.12)',  text: '#2563eb' },
  travel:        { fill: '#06b6d4', border: '#0891b2', bg: 'rgba(6,182,212,0.12)',   text: '#0891b2' },
  base:          { fill: '#94a3b8', border: '#64748b', bg: 'rgba(148,163,184,0.12)', text: '#64748b' },
};

function tc(theme) {
  return THEME[theme] || THEME.base;
}

/* ---- Animation constant ---- */
const TRAJ_MS_PER_SEGMENT = 2000; // ms per node-to-node segment

/* ---- State ---- */
const S = {
  trajectory:         window.__INITIAL_TRAJECTORY__ || null,
  selectedSequenceId: null,
  selectedNodeIndex:  null,
  map:                null,
  markers:            [],   // L.Marker[]
  polyline:           null,
  chatHistoryCache:   {},   // { "seqId:nodeIndex": [{role, content, mode?, emotion?}] }
  customProfile:      null, // overrides trajectory user_profile when set
  _animTimeouts:      [],   // pending animation timeouts (cleared on re-render)
  _animRaf:           null, // requestAnimationFrame handle
  _headMarker:        null, // (unused, kept for cleanup guard)
  segLines:           [],   // per-segment L.polyline instances
  _tlGeoNodes:        [],   // geoNodes used by the current timeline bar
  _tlActiveIdx:       -1,   // which timeline node is currently "active"
};

if (S.trajectory) {
  S.selectedSequenceId = S.trajectory.sequence.sequence_id;
}

/* ---- DOM ---- */
const $ = (id) => document.getElementById(id);

const seqSelect     = $('sequence-select');
const seqMeta       = $('seq-meta');
const profileDemo   = $('profile-demo');
const nodeOverlay   = $('node-overlay');
const ovSeq         = $('ov-seq');
const ovTheme       = $('ov-theme');
const ovNodes       = $('ov-nodes');
const nodeDrawer    = $('node-drawer');
const drawerBadge   = $('drawer-badge');
const drawerTitle   = $('drawer-title');
const drawerSub     = $('drawer-sub');
const drawerClose   = $('drawer-close');
const nodeMeta      = $('node-meta');
const eventCard     = $('event-card');
const motivGrid     = $('motivation-grid');
const chatFab       = $('chat-fab');
const btnOpenChat   = $('btn-open-chat');
const chatModal     = $('chat-modal');
const btnCloseChat  = $('btn-close-chat');
const chatModalTitle = $('chat-modal-title');
const chatMessages  = $('chat-messages');
const chatInput     = $('chat-input');
const sendBtn       = $('send-btn');
const clearBtn      = $('clear-btn');

const btnEditProfile       = $('btn-edit-profile');
const profileEditModal     = $('profile-edit-modal');
const btnCloseProfileEdit  = $('btn-close-profile-edit');
const btnCancelProfileEdit = $('btn-cancel-profile-edit');
const btnSaveProfile       = $('btn-save-profile');
const btnResetProfile      = $('btn-reset-profile');
const profileEditedBadge   = $('profile-edited-badge');

/* ---- Init ---- */
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  bindEvents();
  if (S.trajectory) {
    renderAll(S.trajectory);
  }
  // Re-render dynamic content on language switch
  window.onLangChange = () => {
    if (!S.trajectory) return;
    renderSidebar(S.customProfile || S.trajectory.user_profile, S.trajectory.sequence);
    if (S.selectedNodeIndex !== null) {
      updateOverlayForNode(S.trajectory.nodes[S.selectedNodeIndex]);
    } else {
      resetOverlay(S.trajectory.sequence, S.trajectory.nodes);
    }
  };
});

function bindEvents() {
  seqSelect.addEventListener('change', (e) => loadTrajectory(e.target.value));

  drawerClose.addEventListener('click', closeDrawer);

  // Close drawer when clicking map background (but not when modal is open)
  $('map-area').addEventListener('click', (e) => {
    if (!chatModal.classList.contains('hidden')) return;
    if (e.target === $('map') || e.target.closest('.leaflet-tile-pane')) closeDrawer();
  });

  btnOpenChat.addEventListener('click', openChatModal);
  btnCloseChat.addEventListener('click', closeChatModal);

  // Click outside panel closes modal
  chatModal.addEventListener('click', (e) => {
    if (e.target === chatModal) closeChatModal();
  });

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
    await fetch('/api/clear-session', { method: 'POST' });
    S.selectedNodeIndex = null;
    S.chatHistoryCache = {};
    closeChatModal();
    closeDrawer();
    refreshAllMarkers();
    showToast(t('toast.session_reset'), 'info');
  });
}

/* ---- Map ---- */
function initMap() {
  S.map = L.map('map', { zoomControl: false }).setView([39.9, 116.4], 10);

  // CartoDB Light — cleaner than OSM, better for data overlay
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(S.map);

  L.control.zoom({ position: 'bottomright' }).addTo(S.map);

  // On zoom: resize markers and ensure completed segments have no stale dash styles
  S.map.on('zoomend', () => {
    refreshAllMarkers();
    (S.segLines || []).forEach(seg => {
      const el = seg.getElement();
      if (el) { el.style.strokeDasharray = ''; el.style.strokeDashoffset = ''; }
    });
  });
}

/* ---- Load trajectory ---- */
async function loadTrajectory(seqId) {
  const res  = await fetch(`/api/trajectory/${seqId}`);
  const data = await res.json();
  if (!data.success) {
    showToast(t('toast.load_fail') + (data.error || 'unknown error'), 'error');
    return;
  }
  S.trajectory         = data;
  S.selectedSequenceId = seqId;
  S.selectedNodeIndex  = null;
  S.chatHistoryCache   = {};
  S.customProfile      = null;
  updateProfileEditedBadge();
  closeDrawer();
  renderAll(data);
}

/* ---- Render everything ---- */
function renderAll(traj) {
  renderSidebar(S.customProfile || traj.user_profile, traj.sequence);
  renderOverlay(traj.sequence, traj.nodes);
  renderMap(traj.nodes, traj.sequence.theme);
}

/* ---- Sidebar ---- */
function renderSidebar(profile, sequence) {
  const p  = profile || {};
  const color = tc(sequence.theme);

  // Sequence meta
  seqMeta.innerHTML = `
    <div class="seq-meta-row"><span class="seq-meta-label">${t('sb.theme')}</span>
      <span class="seq-meta-value">
        <span class="tag tag-purple" style="background:${color.bg};color:${color.text};">${esc(sequence.theme || t('misc.na'))}</span>
      </span>
    </div>
    ${sequence.longterm_goal ? `
    <div class="seq-meta-row"><span class="seq-meta-label" style="width:auto;margin-right:6px;">${t('sb.goal')}</span>
      <span class="seq-meta-value" style="font-size:11px;">${esc(truncate(sequence.longterm_goal, 60))}</span>
    </div>` : ''}
  `;

  // Profile
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
        <div class="tag-wrap">
          ${personality.map(tr => `<span class="tag tag-purple">${esc(tr)}</span>`).join('')}
        </div>
      </div>` : ''}

    ${prefs.length ? `
      <div style="margin-top:12px;">
        <div class="sb-section-title" style="font-size:10px;color:var(--text-muted);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:7px;">${t('sb.prefs')}</div>
        <div class="tag-wrap">
          ${prefs.map(tr => `<span class="tag tag-green">${esc(truncate(tr, 50))}</span>`).join('')}
        </div>
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

const USER_NAMES = window.__USER_NAMES__ || {};
function userName(seqId) { return USER_NAMES[seqId] || seqId; }

/* ---- Overlay ---- */
function renderOverlay(sequence, nodes) {
  ovSeq.textContent   = userName(sequence.sequence_id);
  ovTheme.textContent = sequence.theme || 'N/A';
  ovNodes.textContent = `${nodes.length}${t('ov.nodes_count')}`;
}

function updateOverlayForNode(node) {
  ovSeq.textContent   = `N${node.node_index + 1}`;
  ovTheme.textContent = node.time || '—';
  ovNodes.textContent = node.weather || '—';

  // Relabel rows
  const rows = nodeOverlay.querySelectorAll('.ov-label');
  if (rows[0]) rows[0].textContent = t('ov.node');
  if (rows[1]) rows[1].textContent = t('ov.time');
  if (rows[2]) rows[2].textContent = t('ov.weather');
}

function resetOverlay(sequence, nodes) {
  const rows = nodeOverlay.querySelectorAll('.ov-label');
  if (rows[0]) rows[0].textContent = t('ov.user');
  if (rows[1]) rows[1].textContent = t('ov.theme');
  if (rows[2]) rows[2].textContent = t('ov.nodes');
  renderOverlay(sequence, nodes);
}

/* ---- Map rendering ---- */
function renderMap(nodes, theme) {
  // Cancel any in-flight animation
  S._animTimeouts.forEach(t => clearTimeout(t));
  S._animTimeouts = [];
  if (S._animRaf) { cancelAnimationFrame(S._animRaf); S._animRaf = null; }

  // Clear old layers
  S.markers.forEach(m => S.map.removeLayer(m));
  S.markers = [];
  if (S.polyline) { S.map.removeLayer(S.polyline); S.polyline = null; }

  const color = tc(theme);
  const geoNodes = nodes.filter(n =>
    n.location_detail &&
    Number.isFinite(n.location_detail.latitude) &&
    Number.isFinite(n.location_detail.longitude)
  );

  if (!geoNodes.length) {
    $('map-empty').classList.remove('hidden');
    $('timeline-bar').classList.add('hidden');
    return;
  }
  $('map-empty').classList.add('hidden');
  renderTimeline(geoNodes, color);

  const firstLL = [geoNodes[0].location_detail.latitude, geoNodes[0].location_detail.longitude];
  S.map.setView(firstLL, 14);

  S._animTimeouts.push(setTimeout(() => {
    _addNodeMarker(geoNodes[0], color);
    _animateToNode(geoNodes, color, 1);
  }, 380));
}

/* Pan to node[idx], add its marker, pause 1 s, then recurse */
function _animateToNode(geoNodes, color, idx) {
  if (idx >= geoNodes.length) {
    // All nodes shown — zoom out and draw dashed polyline
    _finishWithDashedLine(geoNodes, color);
    return;
  }

  const toNode = geoNodes[idx];
  const toLL   = [toNode.location_detail.latitude, toNode.location_detail.longitude];

  S.map.panTo(toLL, { animate: true, duration: TRAJ_MS_PER_SEGMENT / 1000 });

  S._animTimeouts.push(setTimeout(() => {
    _addNodeMarker(toNode, color);
    S._animTimeouts.push(setTimeout(() => {
      _animateToNode(geoNodes, color, idx + 1);
    }, 1000));
  }, TRAJ_MS_PER_SEGMENT));
}

/* After last node: fit all points into view, then draw dashed polyline */
function _finishWithDashedLine(geoNodes, color) {
  const allLatLngs = geoNodes.map(n => [
    n.location_detail.latitude,
    n.location_detail.longitude,
  ]);

  // Draw polyline, zoom out, and close all popups
  S._animTimeouts.push(setTimeout(() => {
    S.polyline = L.polyline(allLatLngs, {
      color:     color.fill,
      weight:    2.5,
      opacity:   0.55,
      dashArray: '8 6',
      lineCap:   'round',
    }).addTo(S.map);
    S.map.fitBounds(L.latLngBounds(allLatLngs), { padding: [60, 60], animate: true });
    finishTimeline();
    // Close all popups — show only nodes and edges after animation
    S.markers.forEach(m => m.closePopup());
  }, 600));
}

/* Add a node marker with pop-in animation and a permanent popup */
function _addNodeMarker(node, color) {
  // Advance the timeline to this node
  const geoIdx = S._tlGeoNodes.findIndex(n => n.node_index === node.node_index);
  if (geoIdx >= 0) activateTimelineNode(geoIdx);

  const latlng = [node.location_detail.latitude, node.location_detail.longitude];
  // Higher node_index → higher zIndexOffset so newer nodes always appear on top
  const marker = L.marker(latlng, {
    icon:          makeIcon(node.node_index, false, color, true),
    zIndexOffset:  node.node_index * 200,
  })
    .bindPopup(_makeNodePopup(node, color), {
      autoClose:    false,
      closeOnClick: false,
      className:    'node-popup',
      maxWidth:     240,
      offset:       [0, -8],
    })
    .addTo(S.map);

  marker.nodeIndex = node.node_index;
  marker.on('click', () => selectNode(node.node_index));
  S.markers.push(marker);

  // Open popup slightly after marker pop-in, then push it to front
  S._animTimeouts.push(setTimeout(() => {
    marker.openPopup();
    const el = marker.getPopup().getElement();
    if (el) el.style.zIndex = 650 + node.node_index;
  }, 180));
}

function _makeNodePopup(node, color) {
  const rows = [
    node.time     ? `<div class="np-row"><span class="np-icon">🕐</span>${esc(node.time)}</div>`     : '',
    node.location ? `<div class="np-row"><span class="np-icon">📍</span>${esc(node.location)}</div>` : '',
    node.weather  ? `<div class="np-row"><span class="np-icon">🌤</span>${esc(node.weather)}</div>`  : '',
  ].join('');
  const eventText = node.life_event || node.event || '';
  return `
    <div class="np-header" style="background:${color.bg}; color:${color.text};">
      <span class="np-num" style="background:${color.fill};">N${node.node_index + 1}</span>
      <span class="np-title">${esc(truncate(node.intent || eventText, 40))}</span>
    </div>
    <div class="np-meta">${rows}</div>
    ${eventText ? `<div class="np-event">${esc(truncate(eventText, 90))}</div>` : ''}
  `;
}

function markerBaseSize() {
  const zoom = S.map ? S.map.getZoom() : 13;
  // 28px at zoom 13, +4px per zoom level, clamped 16–56
  return Math.max(16, Math.min(112, 28 + (zoom - 13) * 8));
}

function makeIcon(nodeIndex, active, color, animated = false) {
  const base = markerBaseSize();
  const size = active ? Math.round(base * 1.28) : base;
  const bg   = active ? color.fill : '#ffffff';
  const txt  = active ? '#ffffff' : color.fill;
  const ring = active
    ? `0 0 0 3px ${color.fill}55, 0 4px 14px rgba(0,0,0,0.25)`
    : '0 2px 8px rgba(0,0,0,0.18)';
  const fontSize = Math.max(9, Math.round(size * 0.4));

  return L.divIcon({
    className: '',
    html: `<div class="map-marker${animated ? ' map-marker--pop' : ''}" style="
      width:${size}px; height:${size}px;
      background:${bg}; color:${txt};
      border:2.5px solid ${color.fill};
      box-shadow:${ring};
      font-size:${fontSize}px;
    ">${nodeIndex + 1}</div>`,
    iconSize:   [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function refreshAllMarkers() {
  if (!S.trajectory) return;
  const theme = S.trajectory.sequence.theme;
  const color = tc(theme);
  S.markers.forEach(m => {
    const active = m.nodeIndex === S.selectedNodeIndex;
    m.setIcon(makeIcon(m.nodeIndex, active, color));
  });
}

function panToNode(node) {
  if (!node.location_detail) return;
  const { latitude: lat, longitude: lng } = node.location_detail;
  if (Number.isFinite(lat) && Number.isFinite(lng)) {
    S.map.setView([lat, lng], Math.max(S.map.getZoom(), 13), { animate: true });
  }
}

/* ---- Select node ---- */
async function selectNode(nodeIndex) {
  const res  = await fetch('/api/select-node', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ sequence_id: S.selectedSequenceId, node_index: nodeIndex }),
  });
  const data = await res.json();
  if (!data.success) { showToast(t('toast.node_fail'), 'error'); return; }

  S.selectedNodeIndex = nodeIndex;

  // Seed cache from server if we don't have local history yet
  const cacheKey = `${S.selectedSequenceId}:${nodeIndex}`;
  if (!S.chatHistoryCache[cacheKey]) {
    S.chatHistoryCache[cacheKey] = data.chat_history || [];
  }

  // Close modal when switching nodes
  closeChatModal();

  refreshAllMarkers();
  panToNode(data.selected_node);
  updateOverlayForNode(data.selected_node);
  openDrawer(data.selected_node);
}

/* ---- Drawer ---- */
function openDrawer(node) {
  const theme = S.trajectory ? S.trajectory.sequence.theme : 'base';
  const color = tc(theme);

  // Header
  drawerBadge.textContent     = node.node_index + 1;
  drawerBadge.style.background = color.fill;
  drawerTitle.textContent     = `${t('ov.node')} ${node.node_index + 1} · ${esc(theme)}`;
  drawerSub.textContent       = [node.time, node.location].filter(Boolean).join(' · ');

  // Meta
  nodeMeta.innerHTML = `
    ${mrow(t('ov.time'), node.time)}
    ${mrow(t('dr.location'), node.location)}
    ${mrow(t('ov.weather'), node.weather)}
    ${mrow(t('dr.intent'), node.intent)}
  `;

  // Event
  eventCard.textContent = node.event || 'N/A';
  eventCard.style.borderLeftColor = color.fill;

  // Motivation
  const explicit = (node.motivation && node.motivation.explicit) || [];
  const implicit = (node.motivation && node.motivation.implicit) || [];
  motivGrid.innerHTML = `
    <div class="motivation-block mb-explicit">
      <div class="motivation-block-title">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        ${t('dr.explicit')}
      </div>
      ${motivList(explicit)}
    </div>
    <div class="motivation-block mb-implicit">
      <div class="motivation-block-title">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/>
        </svg>
        ${t('dr.implicit')}
      </div>
      ${motivList(implicit)}
    </div>
  `;

  nodeDrawer.classList.add('open');
  chatFab.classList.remove('hidden');
}

function closeDrawer() {
  closeChatModal();
  chatFab.classList.add('hidden');
  nodeDrawer.classList.remove('open');
  if (S.selectedNodeIndex !== null) {
    S.selectedNodeIndex = null;
    refreshAllMarkers();
    if (S.trajectory) {
      resetOverlay(S.trajectory.sequence, S.trajectory.nodes);
    }
  }
}

function mrow(label, value) {
  if (!value) return '';
  return `<span class="node-meta-label">${label}</span><span class="node-meta-value">${esc(value)}</span>`;
}

function motivList(items) {
  if (!items.length) return `<div class="motivation-empty">${t('dr.none')}</div>`;
  return `<ul class="motivation-list">${items.map(i => `<li>${esc(i)}</li>`).join('')}</ul>`;
}

/* ---- Chat modal ---- */
function openChatModal() {
  if (S.selectedNodeIndex === null) return;
  const cacheKey = `${S.selectedSequenceId}:${S.selectedNodeIndex}`;
  const node = S.trajectory && S.trajectory.nodes[S.selectedNodeIndex];
  chatModalTitle.textContent = node
    ? `N${node.node_index + 1} · ${node.location || ''}`
    : t('chat.title');

  const history = S.chatHistoryCache[cacheKey] || [];
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

/* ---- Profile editing ---- */
function effectiveProfile() {
  return S.customProfile || (S.trajectory && S.trajectory.user_profile) || {};
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
  const base = effectiveProfile();
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

  S.customProfile  = updated;
  S.chatHistoryCache = {};
  updateProfileEditedBadge();
  closeProfileEditModal();
  if (S.trajectory) renderSidebar(updated, S.trajectory.sequence);
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

  S.customProfile  = null;
  S.chatHistoryCache = {};
  updateProfileEditedBadge();
  closeProfileEditModal();
  if (S.trajectory) renderSidebar(S.trajectory.user_profile, S.trajectory.sequence);
  showToast(t('toast.profile_reset'), 'info');
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

  // Show emotion tag on assistant bubbles when available
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
  sendBtn.disabled = true;
  const divider = document.createElement('div');
  divider.className = 'chat-ended-notice';
  divider.textContent = t('chat.ended');
  chatMessages.appendChild(divider);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg || S.selectedNodeIndex === null) return;

  const cacheKey = `${S.selectedSequenceId}:${S.selectedNodeIndex}`;
  const cache = S.chatHistoryCache[cacheKey] = S.chatHistoryCache[cacheKey] || [];

  chatInput.value = '';
  chatInput.disabled = true;
  sendBtn.disabled = true;

  appendBubble('user', msg);
  cache.push({ role: 'user', content: msg });

  // Typing indicator
  const typingWrap = document.createElement('div');
  typingWrap.className = 'chat-msg chat-msg-assist';
  typingWrap.innerHTML = `
    <div class="chat-avatar">${t('chat.u')}</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  chatMessages.appendChild(typingWrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const res = await fetch('/api/node-chat', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      sequence_id: S.selectedSequenceId,
      node_index:  S.selectedNodeIndex,
      message:     msg,
    }),
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
      return;  // input stays disabled
    }
  } else {
    typingWrap.remove();
    cache.pop();  // remove the user message we added optimistically
    showToast(t('toast.send_fail') + (data.error || 'unknown'), 'error');
  }

  chatInput.disabled = false;
  sendBtn.disabled   = false;
  chatInput.focus();
}

/* ---- Timeline Bar ---- */

function renderTimeline(geoNodes, color) {
  const bar       = $('timeline-bar');
  const trackWrap = $('tml-track-wrap');
  const fill      = $('tml-fill');

  S._tlGeoNodes  = geoNodes;
  S._tlActiveIdx = -1;

  // Remove previously injected nodes (keep the fill div)
  trackWrap.querySelectorAll('.tml-node').forEach(el => el.remove());
  fill.style.width      = '0%';
  fill.style.background = color.fill;

  if (!geoNodes.length) { bar.classList.add('hidden'); return; }

  const n = geoNodes.length;
  geoNodes.forEach((node, i) => {
    const pct = n === 1 ? 50 : (i / (n - 1)) * 100;

    const nodeEl = document.createElement('div');
    nodeEl.className = 'tml-node';
    nodeEl.style.left = `${pct}%`;
    nodeEl.style.setProperty('--tml-color', color.fill);

    const dot = document.createElement('div');
    dot.className = 'tml-dot';

    const label = document.createElement('div');
    label.className = 'tml-label';
    const tl = nodeTimeLabel(node.time, node.node_index);
    label.innerHTML =
      (tl.year ? `<span class="tml-label-year">${tl.year}</span>` : '') +
      (tl.date ? `<span class="tml-label-date">${tl.date}</span>` : '') +
      `<span class="tml-label-time">${tl.time}</span>`;
    if (node.time) label.title = node.time;

    nodeEl.appendChild(dot);
    nodeEl.appendChild(label);
    nodeEl.addEventListener('click', () => {
      if (nodeEl.classList.contains('tml-node--done') ||
          nodeEl.classList.contains('tml-node--active')) {
        selectNode(node.node_index);
      }
    });
    trackWrap.appendChild(nodeEl);
  });

  bar.classList.remove('hidden');
}

function activateTimelineNode(geoIdx) {
  const trackWrap = $('tml-track-wrap');
  const fill      = $('tml-fill');
  const nodeEls   = trackWrap.querySelectorAll('.tml-node');
  const n         = S._tlGeoNodes.length;

  // Move previous active → done
  if (S._tlActiveIdx >= 0 && nodeEls[S._tlActiveIdx]) {
    nodeEls[S._tlActiveIdx].classList.remove('tml-node--active');
    nodeEls[S._tlActiveIdx].classList.add('tml-node--done');
  }
  S._tlActiveIdx = geoIdx;
  if (nodeEls[geoIdx]) nodeEls[geoIdx].classList.add('tml-node--active');

  // Grow the fill line to this node's position
  const pct = n <= 1 ? 100 : (geoIdx / (n - 1)) * 100;
  fill.style.width = `${pct}%`;
}

function finishTimeline() {
  const trackWrap = $('tml-track-wrap');
  const fill      = $('tml-fill');
  const nodeEls   = trackWrap.querySelectorAll('.tml-node');

  if (S._tlActiveIdx >= 0 && nodeEls[S._tlActiveIdx]) {
    nodeEls[S._tlActiveIdx].classList.remove('tml-node--active');
    nodeEls[S._tlActiveIdx].classList.add('tml-node--done');
  }
  fill.style.width = '100%';
}

function nodeTimeLabel(s, nodeIndex) {
  if (!s) return { year: '', date: '', time: `N${nodeIndex + 1}` };

  // ISO-like: "2024-01-15 08:00" or "2024-01-15T08:00"
  const iso = s.match(/(\d{4})[-/](\d{1,2})[-/](\d{1,2})[T\s](\d{1,2}:\d{2})/);
  if (iso) {
    return { year: iso[1], date: `${iso[2]}/${iso[3]}`, time: iso[4] };
  }

  // Named month: "Jan 15, 2024 08:00" / "January 15 08:00"
  const MONTHS = { jan:1,feb:2,mar:3,apr:4,may:5,jun:6,jul:7,aug:8,sep:9,oct:10,nov:11,dec:12 };
  const named = s.match(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\D+(\d{1,2})[,\s]+(?:(\d{4})[,\s]+)?(\d{1,2}:\d{2})/i);
  if (named) {
    const mon = (MONTHS[named[1].slice(0,3).toLowerCase()] || '?').toString().padStart(2,'0');
    return {
      year: named[3] || '',
      date: `${mon}/${named[2].padStart(2,'0')}`,
      time: named[4],
    };
  }

  // Fallback: pull out whatever time exists
  const t = s.match(/(\d{1,2}:\d{2})/);
  return { year: '', date: s.slice(0, 8).trim(), time: t ? t[1] : s.slice(0, 5) };
}

/* ---- Helpers ---- */
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
