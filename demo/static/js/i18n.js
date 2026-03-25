/* ============================================================
   LifeSim Demo — Internationalisation (EN / ZH)
   Loaded before all page-specific scripts.
   Public API:
     t(key)          — get translated string for current language
     setLang(lang)   — switch language and update the page
   ============================================================ */

const _DICT = {
  en: {
    // Navbar
    'nav.home':   'Home',
    'nav.live':   'Live Generation',
    'nav.preset': 'Preset Demo',

    // Sidebar
    'sb.user':        'User',
    'sb.profile':     'User Profile',
    'sb.reset':       'Reset Session',
    'sb.theme':       'Theme',
    'sb.goal':        'Goal',
    'sb.edited':      'edited',
    'sb.personality': 'Personality',
    'sb.prefs':       'Preferences',

    // Map empty state
    'map.empty':  'No trajectory data',
    'map.select': 'Please select a sequence',

    // Overlay labels
    'ov.user':    'User',
    'ov.theme':   'Theme',
    'ov.nodes':   'Nodes',
    'ov.node':    'Node',
    'ov.time':    'Time',
    'ov.weather': 'Weather',
    'ov.nodes_count': ' nodes',   // prepend number: "5 nodes"

    // Drawer
    'dr.nodeinfo':  'Node Info',
    'dr.happened':  'What Happened',
    'dr.cognitive': 'Cognitive State',
    'dr.explicit':  'Explicit Intents',
    'dr.implicit':  'Implicit Needs',
    'dr.none':      'None',

    // Chat
    'chat.talk':     'Talk to User',
    'chat.title':    'Conversation',
    'chat.empty':    'No messages yet',
    'chat.ph':       'Send a message…',
    'chat.hint':     'Ctrl+Enter to send',
    'chat.ended':    'The user has ended this conversation',
    'chat.fallback': 'Fallback reply (model not connected)',
    'chat.you':      'You',
    'chat.u':        'U',

    // Profile edit modal
    'pe.title':      'Edit User Profile',
    'pe.gender':     'Gender',
    'pe.age':        'Age',
    'pe.area':       'Area',
    'pe.employment': 'Employment',
    'pe.income':     'Income',
    'pe.marital':    'Marital',
    'pe.personality':'Personality',
    'pe.per_hint':   '— one trait per line',
    'pe.prefs':      'Preferences',
    'pe.prefs_hint': '— one item per line',
    'pe.reset':      'Reset to Original',
    'pe.cancel':     'Cancel',
    'pe.save':       'Save & Apply',

    // Toast / status messages
    'toast.session_reset':     'Session reset',
    'toast.load_fail':         'Failed to load: ',
    'toast.node_fail':         'Failed to load node',
    'toast.profile_saved':     'Profile updated — new conversations will use the edited profile',
    'toast.profile_reset':     'Profile reset to original',
    'toast.profile_save_fail': 'Failed to save profile',
    'toast.profile_reset_fail':'Failed to reset profile',
    'toast.send_fail':         'Send failed: ',

    // Drawer node meta labels
    'dr.location': 'Location',
    'dr.intent':   'Intent',

    // Misc
    'misc.loading': 'Loading…',
    'misc.na':      'N/A',

    // ── Index page ──
    'idx.eyebrow':  'Research Demo · Fudan DISC',
    'idx.h1':       'Long-horizon User<br>Life Simulator',
    'idx.desc':     'LifeSim generates coherent long-horizon life trajectories for users, modeling cognitive states (intent + motivation) at each node to capture behavioral variation across contexts.',
    'idx.features': 'Key Features',
    'idx.f1_title': 'Long-horizon Trajectory Generation',
    'idx.f1_desc':  'Generates coherent life event sequences spanning multiple domains, with each node carrying full temporal, spatial, and weather context.',
    'idx.f2_title': 'BDI Cognitive State Modeling',
    'idx.f2_desc':  'Each node carries explicit intents and implicit needs, driving differentiated interaction behavior via the Belief-Desire-Intention model.',
    'idx.f3_title': 'Node-driven Interaction',
    'idx.f3_desc':  'Select any node to chat with the user in that cognitive state. Switch nodes to compare behavioral differences across contexts.',
    'idx.how':      'How It Works',
    'idx.s1_title': 'Select a trajectory',
    'idx.s1_desc':  'Choose a preset sequence from the sidebar — the map instantly renders the full trajectory and user profile.',
    'idx.s2_title': 'Browse the map',
    'idx.s2_desc':  "Follow the user's movement path; semi-transparent dashed lines connect each life node.",
    'idx.s3_title': 'Click a node to inspect cognitive state',
    'idx.s3_desc':  'A detail drawer slides in showing the event description, explicit intents, and implicit needs.',
    'idx.s4_title': 'Chat with the user at that moment',
    'idx.s4_desc':  'Click "Talk to User" to interact with the user in that cognitive state. Switch nodes to contrast behaviors.',

    // ── Live Generation page ──
    'live.select_user':    '— select a user —',
    'live.gen_log':        'Generation Log',
    'live.waiting':        'Waiting for generation…',
    'live.select_begin':   'Select a sequence to begin',
    'live.gen_btn':        'Generate Next Event',
    'live.generating':     'Generating…',
    'live.all_done_btn':   'All events generated',
    'live.all_done_toast': 'All events generated!',
    'live.progress_unit':  ' events',
    'live.empty_title':    'Live Generation Mode',
    'live.empty_sub':      'Select a sequence and click Generate Next Event',
    'live.events_lbl':     'Events',
    'live.calling_llm':    'Calling LLM to generate event…',
    'live.no_result_log':  '✗ No result received.',
    'live.fail_no_result': 'Generation failed: no result received',
    'live.fail_start':     'Failed to start: ',
    'live.fail_gen':       'Generation failed: ',
    'live.done_log':       '✓ Done.',
    'live.fail_node':      'Failed to load node',
  },

  zh: {
    'nav.home':   '首页',
    'nav.live':   '实时生成',
    'nav.preset': '预设演示',

    'sb.user':        '用户',
    'sb.profile':     '用户画像',
    'sb.reset':       '重置会话',
    'sb.theme':       '主题',
    'sb.goal':        '长期目标',
    'sb.edited':      '已修改',
    'sb.personality': '性格特征',
    'sb.prefs':       '偏好',

    'map.empty':  '暂无轨迹数据',
    'map.select': '请选择一个序列',

    'ov.user':    '用户',
    'ov.theme':   '主题',
    'ov.nodes':   '节点数',
    'ov.node':    '节点',
    'ov.time':    '时间',
    'ov.weather': '天气',
    'ov.nodes_count': '个节点',

    'dr.nodeinfo':  '节点信息',
    'dr.happened':  '发生了什么',
    'dr.cognitive': '认知状态',
    'dr.explicit':  '显式意图',
    'dr.implicit':  '隐式需求',
    'dr.none':      '无',

    'chat.talk':     '与用户对话',
    'chat.title':    '对话',
    'chat.empty':    '暂无消息',
    'chat.ph':       '输入消息…',
    'chat.hint':     'Ctrl+Enter 发送',
    'chat.ended':    '用户已结束本次对话',
    'chat.fallback': '备用回复（模型未连接）',
    'chat.you':      '你',
    'chat.u':        '用',

    'pe.title':      '编辑用户画像',
    'pe.gender':     '性别',
    'pe.age':        '年龄',
    'pe.area':       '地区',
    'pe.employment': '就业状态',
    'pe.income':     '收入水平',
    'pe.marital':    '婚姻状况',
    'pe.personality':'性格特征',
    'pe.per_hint':   '— 每行一个特征',
    'pe.prefs':      '偏好',
    'pe.prefs_hint': '— 每行一个偏好',
    'pe.reset':      '恢复原始画像',
    'pe.cancel':     '取消',
    'pe.save':       '保存并应用',

    'toast.session_reset':     '会话已重置',
    'toast.load_fail':         '加载失败：',
    'toast.node_fail':         '加载节点失败',
    'toast.profile_saved':     '画像已更新，新对话将使用修改后的画像',
    'toast.profile_reset':     '画像已恢复原始设置',
    'toast.profile_save_fail': '保存画像失败',
    'toast.profile_reset_fail':'重置画像失败',
    'toast.send_fail':         '发送失败：',

    // Drawer node meta labels
    'dr.location': '位置',
    'dr.intent':   '意图',

    'misc.loading': '加载中…',
    'misc.na':      '无',

    // ── Index page ──
    'idx.eyebrow':  '研究演示 · 复旦 DISC',
    'idx.h1':       '长周期用户<br>生活模拟系统',
    'idx.desc':     'LifeSim 面向长周期用户行为模拟，为每个生活节点建模认知状态（意图与需求），捕捉用户在不同情境下的行为差异。',
    'idx.features': '核心功能',
    'idx.f1_title': '长周期轨迹生成',
    'idx.f1_desc':  '生成跨多领域的连贯生活事件序列，每个节点携带完整的时间、空间与天气上下文。',
    'idx.f2_title': 'BDI 认知状态建模',
    'idx.f2_desc':  '每个节点包含显式意图和隐式需求，通过信念-愿望-意图模型驱动差异化的交互行为。',
    'idx.f3_title': '节点驱动交互',
    'idx.f3_desc':  '选择任意节点，与处于该认知状态下的用户对话。切换节点可对比不同情境下的行为差异。',
    'idx.how':      '工作原理',
    'idx.s1_title': '选择轨迹',
    'idx.s1_desc':  '从侧边栏选择预设序列——地图立即渲染完整轨迹和用户画像。',
    'idx.s2_title': '浏览地图',
    'idx.s2_desc':  '沿用户移动路径查看，半透明虚线连接各个生活节点。',
    'idx.s3_title': '点击节点查看认知状态',
    'idx.s3_desc':  '详细面板滑出，显示事件描述、显式意图和隐式需求。',
    'idx.s4_title': '与该时刻的用户对话',
    'idx.s4_desc':  '点击"与用户对话"，与该认知状态下的用户互动；切换节点，对比行为差异。',

    // ── Live Generation page ──
    'live.select_user':    '— 选择用户 —',
    'live.gen_log':        '生成日志',
    'live.waiting':        '等待生成…',
    'live.select_begin':   '请选择序列开始',
    'live.gen_btn':        '生成下一个事件',
    'live.generating':     '生成中…',
    'live.all_done_btn':   '所有事件已生成',
    'live.all_done_toast': '所有事件已生成！',
    'live.progress_unit':  '个事件',
    'live.empty_title':    '实时生成模式',
    'live.empty_sub':      '选择序列后点击"生成下一个事件"',
    'live.events_lbl':     '事件数',
    'live.calling_llm':    '调用 LLM 生成事件…',
    'live.no_result_log':  '✗ 未收到结果。',
    'live.fail_no_result': '生成失败：未收到结果',
    'live.fail_start':     '启动失败：',
    'live.fail_gen':       '生成失败：',
    'live.done_log':       '✓ 完成。',
    'live.fail_node':      '加载节点失败',
  },
};

/* ---- Core ---- */

let _lang = localStorage.getItem('lifesim_lang') || 'en';

function t(key) {
  return (_DICT[_lang] && _DICT[_lang][key]) || (_DICT.en && _DICT.en[key]) || key;
}

function setLang(lang) {
  _lang = lang;
  localStorage.setItem('lifesim_lang', lang);
  applyTranslations();
  // Let page-specific JS re-render any dynamic content
  if (typeof window.onLangChange === 'function') window.onLangChange();
}

function applyTranslations() {
  // Static HTML elements annotated with data-i18n
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.getAttribute('data-i18n'));
  });
  // Elements that need innerHTML (e.g. contain <br>)
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    el.innerHTML = t(el.getAttribute('data-i18n-html'));
  });
  // Input/textarea placeholders
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    el.placeholder = t(el.getAttribute('data-i18n-ph'));
  });
  // Toggle button: show the OTHER language as the label
  const btn = document.getElementById('lang-toggle');
  if (btn) btn.textContent = _lang === 'en' ? '中文' : 'EN';
  // <html lang="…">
  document.documentElement.lang = _lang === 'zh' ? 'zh' : 'en';
}

document.addEventListener('DOMContentLoaded', () => {
  applyTranslations();
  const btn = document.getElementById('lang-toggle');
  if (btn) btn.addEventListener('click', () => setLang(_lang === 'en' ? 'zh' : 'en'));
});
