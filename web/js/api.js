/* API 调用封装 + Toast 通知 */

const API_BASE = window.location.origin;

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
    const resp = await fetch(`${API_BASE}${path}`, opts);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0,200)}`);
    return resp.json();
  }

  static transcribe(file, params = {}) {
    const fd = new FormData(); fd.append('file', file);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.request('/v1/audio/transcriptions', { method:'POST', body:fd });
  }

  static submitTask(fileOrUrl, params = {}) {
    const fd = new FormData();
    if (fileOrUrl instanceof File) fd.append('file', fileOrUrl);
    else fd.append('url', fileOrUrl);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.request('/api/tasks/submit', { method:'POST', body:fd });
  }

  static getTask(taskId) { return this.request(`/api/tasks/${taskId}`); }
  static listTasks() { return this.request('/api/tasks'); }
  static deleteTask(taskId) { return this.request(`/api/tasks/${taskId}`, { method:'DELETE' }); }

  static recognition(file, params = {}) {
    const fd = new FormData(); fd.append('audio', file);
    Object.entries(params).forEach(([k,v]) => { if (v!==undefined && v!==null && v!=='') fd.append(k, v); });
    return this.request('/recognition', { method:'POST', body:fd });
  }

  static health() { return this.request('/health'); }

  // 声纹
  static registerSpeaker(audioFile, name, group=null) {
    const fd = new FormData(); fd.append('audio', audioFile); fd.append('name', name);
    if (group) fd.append('speaker_group', group);
    return this.request('/api/speakers/register', { method:'POST', body:fd });
  }
  static listSpeakerGroups() { return this.request('/api/speakers'); }
  static getSpeakers(groupId) { return this.request(`/api/speakers/${groupId}`); }
  static deleteSpeaker(groupId, name) { return this.request(`/api/speakers/${groupId}/${encodeURIComponent(name)}`, { method:'DELETE' }); }
}
