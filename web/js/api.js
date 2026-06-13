/* API 调用封装 + Toast + Token 管理 */

// 兼容 file:// 协议
const API_BASE = (window.location.protocol === 'file:' || window.location.origin === 'null')
  ? 'http://localhost:17767'
  : window.location.origin;

// ── Token 管理 ──────────────────────────────────
function getToken() { return localStorage.getItem('funasr_token') || ''; }
function setToken(t) { localStorage.setItem('funasr_token', t); }

function promptToken() {
  const current = getToken();
  const token = window.prompt('请输入 API Token（服务端 .env 中 API_TOKEN 的值）：', current);
  if (token !== null) {
    setToken(token.trim());
    toast(token.trim() ? '🔑 Token 已保存' : 'Token 已清除，刷新页面生效', 'success');
  }
}

// 页面加载后注入 Token 按钮到导航
document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('.nav');
  if (!nav) return;
  const a = document.createElement('a');
  a.href = '#';
  a.textContent = getToken() ? '🔑 已认证' : '🔑 Token';
  a.onclick = (e) => { e.preventDefault(); promptToken(); a.textContent = getToken() ? '🔑 已认证' : '🔑 Token'; };
  a.style.marginLeft = 'auto';
  nav.appendChild(a);
});

// ── Toast ────────────────────────────────────────
function toast(msg, type = 'info', duration = 3000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, duration);
}

// ── 格式化 ────────────────────────────────────────
function formatBytes(b) { return b < 1024*1024 ? (b/1024).toFixed(1)+'KB' : (b/1024/1024).toFixed(1)+'MB'; }
function formatDuration(ms) { const s = ms/1000; return s < 60 ? s.toFixed(1)+'s' : Math.floor(s/60)+'m'+Math.round(s%60)+'s'; }
function formatTime(ts) { return ts ? new Date(ts*1000).toLocaleString('zh-CN') : '-'; }
function statusLabel(s) { return {downloading:'下载中',queued:'排队中',processing:'处理中',completed:'已完成',failed:'失败'}[s]||s; }
function speakerColor(n) { return `spk-${(n||0)%5}`; }

// ── 剪贴板 ────────────────────────────────────────
function copyText(text) {
  navigator.clipboard.writeText(text).then(() => toast('已复制到剪贴板','success')).catch(() => toast('复制失败','error'));
}

// ── API ──────────────────────────────────────────
class FunASRApi {
  static async request(path, opts = {}) {
    const token = getToken();
    if (token) {
      opts.headers = { ...(opts.headers || {}), 'Authorization': `Bearer ${token}` };
    }
    const resp = await fetch(`${API_BASE}${path}`, opts);
    if (resp.status === 401) { toast('🔑 Token 无效或缺失，请点击导航栏 Token 按钮设置', 'error', 5000); }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0,200)}`);
    return resp.json();
  }

  static transcribe(file, params = {}) {
    const fd = new FormData(); fd.append('file', file);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.requestForm('/api/v1/transcriptions', fd, params.response_format || 'json');
  }

  static async transcribeOpenAI(file, params = {}) {
    const fd = new FormData(); fd.append('file', file);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.requestForm('/v1/audio/transcriptions', fd, params.response_format || 'json');
  }

  static async requestForm(path, formData, responseFormat = 'json') {
    const token = getToken();
    const headers = token ? { 'Authorization': `Bearer ${token}` } : undefined;
    const resp = await fetch(`${API_BASE}${path}`, { method:'POST', body:formData, headers });
    if (resp.status === 401) { toast('🔑 Token 无效或缺失，请点击导航栏 Token 按钮设置', 'error', 5000); }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0,200)}`);
    const contentType = resp.headers.get('content-type') || '';
    if (contentType.includes('application/json')) return resp.json();
    return { text: await resp.text(), response_format: responseFormat };
  }

  static submitTask(fileOrUrl, params = {}) {
    const fd = new FormData();
    if (fileOrUrl instanceof File) fd.append('file', fileOrUrl);
    else fd.append('url', fileOrUrl);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.request('/api/v1/transcription-jobs', { method:'POST', body:fd });
  }

  static getTask(taskId) { return this.request(`/api/v1/transcription-jobs/${taskId}`); }
  static listTasks() { return this.request('/api/v1/transcription-jobs'); }
  static deleteTask(taskId) { return this.request(`/api/v1/transcription-jobs/${taskId}`, { method:'DELETE' }); }

  static health() { return this.request('/health'); }

  // 声纹
  static async registerSpeaker(audioFile, name, group=null) {
    let targetGroup = group;
    if (!targetGroup) {
      const created = await this.request('/api/v1/speaker-groups', { method:'POST' });
      targetGroup = created.group_id;
    }
    const fd = new FormData(); fd.append('audio', audioFile); fd.append('name', name);
    const path = `/api/v1/speaker-groups/${encodeURIComponent(targetGroup)}/speakers`;
    return this.request(path, { method:'POST', body:fd });
  }
  static listSpeakerGroups() { return this.request('/api/v1/speaker-groups'); }
  static getSpeakers(groupId) { return this.request(`/api/v1/speaker-groups/${groupId}/speakers`); }
  static deleteSpeaker(groupId, name) { return this.request(`/api/v1/speaker-groups/${groupId}/speakers/${encodeURIComponent(name)}`, { method:'DELETE' }); }
}

// ── 页面初始化（标题显示模型名 + 连接状态）──────────
async function initPage(pageName) {
  try {
    const data = await FunASRApi.health();
    document.title = `🎙️ ${data.model || 'FunASR'} · ${pageName}`;
    return data;
  } catch(e) {
    document.title = `❌ FunASR (连接失败) · ${pageName}`;
    return null;
  }
}
