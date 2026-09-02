"""
FunASR Web Service with Speaker Diarization

Features:
  - ASR: SenseVoice, Paraformer (CPU optimized)
  - Speaker Diarization: CAM++ (lazy-loaded on first spk=true request)
  - Multi-worker support
  - Web upload UI
  - File storage & management (list, delete, download)
  - API documentation on web

Usage:
    python server.py --device cpu --port 8000 --workers 2
"""

import argparse
import io
import os
import re
import time
import json
import uuid
import shutil
import subprocess
import tempfile
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="FunASR Web Service", version="1.0.0",
              description="语音识别 + 说话人分离服务。上传音频文件进行转写，返回带说话人标签的分段结果。")

MODEL_REGISTRY = {}
DEVICE = "cpu"
DEFAULT_MODEL = "sensevoice"
N8N_OPENAI_MODEL_ALIAS = "whisper-1"

# Access token from env (optional). When set, all endpoints except
# /health, /docs, /redoc, /openapi.json require "Authorization: Bearer <token>".
API_TOKEN = os.environ.get("FUNASR_TOKEN", "").strip()

# Public paths that skip token auth
PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

# Storage directories
DATA_DIR = os.environ.get("FUNASR_DATA_DIR", "/data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RECORD_DIR = os.path.join(DATA_DIR, "records")
RESULT_DIR = os.path.join(DATA_DIR, "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

for d in (DATA_DIR, AUDIO_DIR, RECORD_DIR, RESULT_DIR, STATIC_DIR):
    os.makedirs(d, exist_ok=True)

JSON_EXT = (".json", ".jsonl")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".amr", ".aac", ".opus", ".wma")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".ts", ".mpg", ".mpeg", ".3gp", ".m4v")

BROWSER_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FunASR 语音识别服务</title>
<style>
:root { --primary: #4f46e5; --primary-dark: #4338ca; --bg: #f4f5f9; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; background: var(--bg); color: #1f2937; }
.header { background: #fff; border-bottom: 1px solid #e5e7eb; padding: 16px 40px; display: flex; align-items: center; gap: 12px; }
.header h1 { font-size: 20px; margin: 0; color: #111827; }
.header .badge { background: #eef2ff; color: var(--primary); font-size: 12px; padding: 4px 10px; border-radius: 20px; font-weight: 600; }
.container { max-width: 1000px; margin: 32px auto; padding: 0 20px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.card h2 { margin-top: 0; font-size: 17px; }
.drop-zone { border: 2px dashed #d1d5db; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: all .2s; }
.drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: #f5f3ff; }
.drop-zone .icon { font-size: 40px; margin-bottom: 8px; }
.drop-zone p { margin: 4px 0; color: #6b7280; }
.model-select { margin-top: 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.model-select label { font-size: 14px; }
select, button { padding: 8px 14px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 14px; background: #fff; }
button.primary { background: var(--primary); color: #fff; border: none; cursor: pointer; font-weight: 600; }
button.primary:hover { background: var(--primary-dark); }
button.danger { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; cursor: pointer; }
.toggle { display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 14px; }
.toggle input { width: 18px; height: 18px; }
.result-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-top: 0; white-space: pre-wrap; font-size: 14px; line-height: 1.8; max-height: 480px; overflow-y: auto; }
.result-toolbar { display:flex; align-items:center; gap:10px; margin-top:16px; padding:8px 12px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px 8px 0 0; font-size:13px; color:#166534; }
.result-toolbar a { color:var(--primary); font-weight:600; text-decoration:none; padding:5px 12px; border-radius:6px; border:1px solid var(--primary); background:#eef2ff; }
.seg { margin-bottom: 10px; }
.seg .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px; color: #fff; }
.spk0 { background: #4f46e5; }
.spk1 { background: #db2777; }
.spk2 { background: #059669; }
.spk3 { background: #d97706; }
.spk4 { background: #0891b2; }
.seg .time { font-size: 12px; color: #9ca3af; margin-right: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #f3f4f6; }
th { color: #6b7280; font-weight: 600; font-size: 12px; text-transform: uppercase; }
tr:hover td { background: #f9fafb; }
.actions button { margin-right: 6px; font-size: 13px; padding: 5px 10px; }
.empty { color: #9ca3af; text-align: center; padding: 20px; }
.tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.tab { padding: 8px 18px; border-radius: 8px; background: #e5e7eb; color: #374151; cursor: pointer; border: none; font-size: 14px; }
.tab.active { background: var(--primary); color: #fff; }
.panel { display: none; }
.panel.active { display: block; }
.loading { opacity: .6; pointer-events: none; }
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin .8s linear infinite; vertical-align: middle; margin-right: 8px; }
@keyframes spin { to { transform: rotate(360deg); } }
.api-links a { color: var(--primary); margin-right: 16px; text-decoration: none; font-size: 14px; }
.spk-color { display:inline-block; width:10px;height:10px;border-radius:50%;margin-right:6px; }
</style>
</head>
<body>
<div class="header">
  <h1>🎙 FunASR 语音识别服务</h1>
  <span class="badge">SenseVoice + CAM++ 说话人分离</span>
  <span style="flex:1"></span>
  <div class="api-links">
    <a href="/docs" target="_blank">接口文档</a>
    <a href="/redoc" target="_blank">ReDoc</a>
    <button onclick="showTokenModal()" style="font-size:12px;padding:4px 10px;margin-left:8px">🔑</button>
  </div>
</div>

  <div class="container">
  <!-- Token Modal -->
  <div id="tokenModal" style="position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:999;display:none;align-items:center;justify-content:center">
    <div style="background:#fff;border-radius:12px;padding:32px;width:360px;text-align:center">
      <h3 style="margin:0 0 12px">请输入访问令牌</h3>
      <input id="tokenInput" type="password" placeholder="Token" onkeydown="if(event.key==='Enter')saveToken()" style="width:100%;padding:10px 12px;border:1px solid #d1d5db;border-radius:8px;font-size:14px;box-sizing:border-box">
      <div style="margin-top:14px">
        <button class="primary" onclick="saveToken()" style="width:100%">确认</button>
      </div>
      <div id="tokenErr" style="color:#dc2626;font-size:12px;margin-top:8px"></div>
    </div>
  </div>

    <div class="tabs">
    <button class="tab active" data-tab="upload">上传识别</button>
    <button class="tab" data-tab="records">历史记录</button>
  </div>

  <!-- Upload Panel -->
  <div class="panel active" id="panel-upload">
    <div class="card">
      <h2>上传音频文件</h2>
      <div class="drop-zone" id="dropZone">
        <div class="icon">📁</div>
        <p><strong>点击选择 或 拖拽文件到此处</strong></p>
        <p>支持格式：WAV, MP3, FLAC, M4A, OGG, WEBM, AAC, OPUS 等</p>
      </div>
      <input type="file" id="fileInput" accept="audio/*" style="display:none" multiple>
      <div class="model-select">
        <label>识别模型：
          <select id="modelSelect">
            <option value="sensevoice" selected>SenseVoice（多语言，推荐）</option>
            <option value="paraformer">Paraformer（中文）</option>
          </select>
        </label>
        <label class="toggle"><input type="checkbox" id="spkToggle" checked> 说话人分离</label>
        <span style="flex:1"></span>
        <button class="primary" id="uploadBtn">开始识别</button>
      </div>
      <div class="result-toolbar" id="resultToolbar" style="display:none">
        <span id="resultSummary"></span>
        <span style="flex:1"></span>
        <a id="exportMdBtn" href="#" onclick="return false" style="pointer-events:none;opacity:.4">导出MD</a>
        <a id="exportJsonBtn" href="#" onclick="return false" style="pointer-events:none;opacity:.4">下载结果</a>
      </div>
      <div class="result-box" id="resultBox" style="display:none"></div>
    </div>
  </div>

  <!-- Records Panel -->
  <div class="panel" id="panel-records">
    <div class="card">
      <h2>历史识别记录</h2>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>文件名</th><th>时长</th><th>说话人</th><th>创建时间</th><th>操作</th></tr></thead>
          <tbody id="recordsBody"><tr><td colspan="5" class="empty">加载中...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let selectedFiles = [];
let lastRecordId = null;
const spkColors = ['#4f46e5','#db2777','#059669','#d97706','#0891b2'];

function getToken() { return localStorage.getItem('funasr_token') || ''; }
function hasToken() { return !!getToken(); }

function showTokenModal() {
  const modal = document.getElementById('tokenModal');
  modal.style.display = 'flex';
  const input = document.getElementById('tokenInput');
  input.value = getToken();
  document.getElementById('tokenErr').textContent = '';
  input.focus();
}
function hideTokenModal() {
  document.getElementById('tokenModal').style.display = 'none';
}
function saveToken() {
  const val = document.getElementById('tokenInput').value.trim();
  if (!val) { document.getElementById('tokenErr').textContent = '请输入令牌'; return; }
  localStorage.setItem('funasr_token', val);
  hideTokenModal();
  boot();
}

async function authFetch(url, opts = {}) {
  const token = getToken();
  if (token) {
    if (!opts.headers) opts.headers = {};
    if (opts.headers instanceof Headers) opts.headers.set('Authorization', 'Bearer ' + token);
    else if (Array.isArray(opts.headers)) opts.headers.push(['Authorization', 'Bearer ' + token]);
    else opts.headers['Authorization'] = 'Bearer ' + token;
  }
  return fetch(url, opts);
}

async function boot() {
  try {
    const r = await authFetch('/api/records');
    if (r.status === 401) { showTokenModal(); return; }
    document.getElementById('tokenModal').style.display = 'none';
  } catch (e) { showTokenModal(); return; }
  initApp();
}

function initApp() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) { selectedFiles = [...e.dataTransfer.files]; updateDropText(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) { selectedFiles = [...fileInput.files]; updateDropText(); }
  });

  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.getElementById('panel-' + t.dataset.tab).classList.add('active');
      if (t.dataset.tab === 'records') loadRecords();
    });
  });

  document.getElementById('uploadBtn').addEventListener('click', handleUpload);
  loadRecords();
}

function updateDropText() {
  const names = selectedFiles.map(f => f.name).join(', ');
  dropZone.querySelector('p strong').textContent = names || '点击选择 或 拖拽文件到此处';
}

async function handleUpload() {
  if (!selectedFiles.length) { alert('请先选择音频文件'); return; }
  const btn = document.getElementById('uploadBtn');
  const box = document.getElementById('resultBox');
  const withSpk = document.getElementById('spkToggle').checked;
  const model = document.getElementById('modelSelect').value;
  btn.disabled = true; btn.classList.add('loading');
  btn.innerHTML = '<span class="spinner"></span>识别中...';
  box.style.display = 'block';
  box.innerHTML = '处理中，请稍候...（说话人分离会稍慢）';

  const results = [];
  try {
    for (const file of selectedFiles) {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('model', model);
      fd.append('spk', withSpk ? 'true' : 'false');
      fd.append('response_format', 'verbose_json');
      const resp = await authFetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.text();
        throw new Error(resp.status === 401 ? '认证失败，请刷新页面重新输入令牌' : err);
      }
      const data = await resp.json();
      results.push({ name: file.name, data });
    }
  } catch (e) {
    box.innerHTML = '<div style="color:#dc2626">识别失败：' + e.message + '</div>';
    btn.disabled = false; btn.classList.remove('loading');
    btn.innerHTML = '开始识别';
    return;
  }

  let html = '';
  results.forEach((r) => {
    html += '<div style="margin-bottom:24px"><h3 style="font-size:14px;margin:8px 0">📄 ' + r.name + '</h3>';
    html += '<div style="font-size:12px;color:#6b7280;margin-bottom:8px">音频时长: ' + (r.data.duration||'') + 's · 识别耗时: ' + (r.data.processing_time||'') + 's · 模型: ' + (r.data.model||'') + '</div>';
    const segs = r.data.segments || [];
    if (segs.length) {
      segs.forEach(s => {
        const spk = s.speaker;
        const color = spk ? spkColors[parseInt(spk.replace('SPK','')) % spkColors.length] : '#9ca3af';
        const label = spk || '未知';
        html += '<div class="seg"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+color+';margin-right:8px"></span>';
        html += '<span class="badge" style="background:'+color+'">'+label+'</span>';
        html += '<span class="time">'+s.start.toFixed(1)+'s - '+s.end.toFixed(1)+'s</span>';
        html += s.text + '</div>';
      });
    } else {
      html += '<div>' + r.data.text + '</div>';
    }
    html += '</div>';
  });
  box.innerHTML = html;

  // 显示工具栏，绑定导出按钮
  if (results.length && results[0].data.record_id) {
    lastRecordId = results[0].data.record_id;
    const tid = lastRecordId;
    document.getElementById('resultSummary').textContent = '识别完成 · 共 ' + results.reduce((a,r)=> a+(r.data.segments||[]).length, 0) + ' 段';
    const mdBtn = document.getElementById('exportMdBtn');
    mdBtn.href = '/api/records/' + tid + '/md'; mdBtn.style.pointerEvents='auto'; mdBtn.style.opacity='1';
    const jBtn = document.getElementById('exportJsonBtn');
    jBtn.href = '/api/records/' + tid + '/result'; jBtn.style.pointerEvents='auto'; jBtn.style.opacity='1';
    document.getElementById('resultToolbar').style.display = 'flex';
  }
  btn.disabled = false; btn.classList.remove('loading');
  btn.innerHTML = '开始识别';
}

async function loadRecords() {
  const body = document.getElementById('recordsBody');
  try {
    const resp = await authFetch('/api/records');
    if (resp.status === 401) { showTokenModal(); return; }
    const records = await resp.json();
    if (!records.length) { body.innerHTML = '<tr><td colspan="5" class="empty">暂无记录</td></tr>'; return; }
    let html = '';
    records.forEach(r => {
      const spk = r.speaker_list && r.speaker_list.length ? r.speaker_list.join(', ') : '—';
      html += '<tr>';
      html += '<td>'+r.filename+'</td>';
      html += '<td>'+r.duration+'s</td>';
      html += '<td>'+spk+'</td>';
      html += '<td>'+r.created+'</td>';
      html += '<td class="actions">';
      html += '<button onclick="viewRecord(\\''+r.id+'\\')">查看</button>';
      html += '<button onclick="downloadFile(\\''+r.id+'\\',\\'audio\\')">下载原文件</button>';
      html += '<button onclick="downloadFile(\\''+r.id+'\\',\\'result\\')">下载结果</button>';
      html += '<button onclick="downloadFile(\\''+r.id+'\\',\\'md\\')">导出MD</button>';
      html += '<button class="danger" onclick="deleteRecord(\\''+r.id+'\\')">删除</button>';
      html += '</td></tr>';
    });
    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = '<tr><td colspan="5" class="empty">加载失败</td></tr>';
  }
}

async function viewRecord(id) {
  const resp = await authFetch('/api/records/' + id);
  if (!resp.ok) { alert('获取失败'); return; }
  const r = await resp.json();
  let html = '<h3 style="font-size:14px">'+r.filename+'</h3>';
  const segs = r.segments || [];
  if (segs.length) {
    segs.forEach(s => {
      const spk = s.speaker;
      const color = spk ? spkColors[parseInt(spk.replace('SPK','')) % spkColors.length] : '#9ca3af';
      html += '<div class="seg"><span class="badge" style="background:'+color+'">'+ (spk||'未知') +'</span>';
      html += '<span class="time">'+s.start.toFixed(1)+'s - '+s.end.toFixed(1)+'s</span>'+s.text+'</div>';
    });
  } else {
    html += '<div>'+r.text+'</div>';
  }
  document.getElementById('resultBox').style.display = 'block';
  document.getElementById('resultBox').innerHTML = html;
  document.querySelector('[data-tab="upload"]').click();
}

async function downloadFile(id, type) {
  const resp = await authFetch('/api/records/'+id+'/'+type);
  if (!resp.ok) { alert('下载失败'); return; }
  const blob = await resp.blob();
  const disp = resp.headers.get('content-disposition') || '';
  const nameMatch = disp.match(/filename=([^;]+)/);
  const filename = nameMatch ? nameMatch[1] : (type === 'audio' ? id+'.wav' : type === 'md' ? id+'.md' : id+'.json');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 100);
}

async function deleteRecord(id) {
  if (!confirm('确认删除该记录和原文件？')) return;
  const resp = await authFetch('/api/records/' + id, { method: 'DELETE' });
  if (resp.ok) loadRecords(); else alert('删除失败');
}

boot();
</script>
</body>
</html>
"""


# ============= Storage Helpers =============

def gen_id():
    return uuid.uuid4().hex[:12]


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_records():
    records = []
    if os.path.isdir(RECORD_DIR):
        for fn in os.listdir(RECORD_DIR):
            if fn.endswith(".json"):
                rec = load_json(os.path.join(RECORD_DIR, fn))
                if rec:
                    records.append(rec)
    records.sort(key=lambda r: r.get("created_ts", 0), reverse=True)
    return records


def find_record(record_id):
    path = os.path.join(RECORD_DIR, f"{record_id}.json")
    return load_json(path), path


# ============= Model Helpers =============

MODEL_CONFIGS = {
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "punc_model": "ct-punc",
    },
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
    },
}

GEN_KWARGS = {
    "batch_size_s": 60,
    "merge_vad": True,
    "merge_length_s": 15,
    "use_itn": True,
}


def load_model(model_name: str, with_spk: bool = False):
    key = f"{model_name}__spk" if with_spk else model_name
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}")

    from funasr import AutoModel
    cfg = MODEL_CONFIGS[model_name].copy()
    cfg["device"] = DEVICE
    cfg["disable_update"] = True
    if with_spk:
        cfg["spk_model"] = "cam++"

    logger.info(f"Loading ASR model '{model_name}' (spk={'on' if with_spk else 'off'}) on {DEVICE}...")
    t0 = time.time()
    model = AutoModel(**cfg)
    logger.info(f"Model '{model_name}' loaded in {time.time() - t0:.1f}s")
    MODEL_REGISTRY[key] = model
    return model


def clean_text(text: str) -> str:
    return re.sub(r'<\|[^|]*\|>', '', text).strip()


def resolve_model(requested: str) -> str:
    if requested == N8N_OPENAI_MODEL_ALIAS:
        return DEFAULT_MODEL
    return requested


def get_bearer_header(authorization: Optional[str] = Header(None)):
    return authorization


def require_token(authorization: str = Depends(get_bearer_header)):
    """FastAPI dependency. Enforces API_TOKEN if configured (except public paths)."""
    if not API_TOKEN:
        return True
    if not authorization:
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    expected = API_TOKEN
    if authorization == f"Bearer {expected}" or authorization == expected or authorization == f"Token {expected}":
        return True
    raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})


# ============= API Endpoints =============

@app.get("/", response_class=HTMLResponse)
async def index():
    return BROWSER_HTML


@app.get("/health")
async def health():
    loaded = sorted({k.replace("__spk", "") for k in MODEL_REGISTRY})
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": loaded,
        "models_available": list(MODEL_CONFIGS.keys()),
    }


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "funasr"}
            for name in MODEL_CONFIGS
        ],
    })


@app.post("/v1/audio/transcriptions")
async def transcribe(
    _: bool = Depends(require_token),
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    spk: bool = Form(default=False),
):
    """OpenAI 兼容转写接口。上传音频，返回文本 / 分段（含说话人标签）。

    参数:
      - file: 音频文件 (wav, mp3, flac, m4a, ogg, webm 等)
      - model: sensevoice 或 paraformer
      - language: 可选语言提示
      - response_format: json 或 verbose_json
      - spk: true 时启用说话人分离

    返回 (verbose_json):
      - text: 完整文本
      - segments: 分段数组 [{start, end, text, speaker}]
      - duration, model
    """
    model = resolve_model(model)
    if model not in MODEL_CONFIGS:
        raise HTTPException(400, f"Model '{model}' not found. Available: {list(MODEL_CONFIGS.keys())}")

    content = await file.read()
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    ext = suffix.lower()
    is_video = ext in VIDEO_EXTS

    # Store original file (audio 或 视频文件)
    record_id = gen_id()
    orig_ext = ext if (ext in AUDIO_EXTS or is_video) else ".wav"
    audio_filename = f"{record_id}{orig_ext}"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(content)

    try:
        # 模型输入：视频等格式先用 ffmpeg 转为 16k 单声道 wav，其余直接保存为临时文件
        tmp_path = ""
        try:
            if is_video:
                tmp_path = os.path.join(tempfile.gettempdir(), f"{record_id}.wav")
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-v", "error", "-i", audio_path,
                        "-vn", "-ac", "1", "-ar", "16000",
                        "-f", "wav", tmp_path,
                    ],
                    check=True,
                )
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
        except Exception as e:
            logger.error(f"Audio prepare error: {e}")
            raise HTTPException(400, f"无法处理音频/视频文件: {e}")

        # Use a model instance that includes CAM++ speaker model when spk requested
        asr_model = load_model(model, with_spk=spk)
        t0 = time.time()

        gen_kwargs = {"input": tmp_path, "batch_size": 1}
        gen_kwargs.update(GEN_KWARGS)
        if language:
            gen_kwargs["language"] = language

        result = asr_model.generate(**gen_kwargs)
        elapsed = time.time() - t0
        text = clean_text(result[0]["text"])

        segments = []
        if "sentence_info" in result[0]:
            for seg in result[0]["sentence_info"]:
                item = {
                    "start": seg.get("start", 0) / 1000.0,
                    "end": seg.get("end", 0) / 1000.0,
                    "text": clean_text(seg.get("text", "")),
                    "speaker": seg.get("spk", None),
                }
                if spk:
                    item["speaker"] = f"SPK{item['speaker']}" if item["speaker"] is not None else None
                else:
                    item.pop("speaker")
                segments.append(item)

            if spk:
                # Renumber speakers by first-appearance order (SPK0, SPK1, ...)
                seen = {}
                for seg in segments:
                    s = seg.get("speaker")
                    if s is None:
                        continue
                    if s not in seen:
                        seen[s] = f"SPK{len(seen)}"
                    seg["speaker"] = seen[s]
        elif spk:
            # spk requested but no sentence_info: try to fallback via verbose segments
            pass

        duration = 0
        try:
            duration = round(float(sf.info(tmp_path).duration), 3)
        except Exception:
            pass

        # Save record
        speaker_list = sorted(set(s["speaker"] for s in segments if s.get("speaker")))
        record = {
            "id": record_id,
            "filename": file.filename or audio_filename,
            "audio_file": audio_filename,
            "created_ts": time.time(),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration,
            "model": model,
            "processing_time": round(elapsed, 3),
            "language": language or "auto",
            "text": text,
            "segments": segments,
            "speaker_list": speaker_list,
        }
        save_json(os.path.join(RECORD_DIR, f"{record_id}.json"), record)
        save_json(os.path.join(RESULT_DIR, f"{record_id}.json"), record)

        if response_format == "verbose_json":
            return JSONResponse({
                "text": text,
                "segments": segments,
                "language": record["language"],
                "duration": duration,
                "processing_time": round(elapsed, 3),
                "model": model,
                "record_id": record_id,
            })
        else:
            return JSONResponse({"text": text, "record_id": record_id})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/records")
async def list_records_api(_: bool = Depends(require_token)):
    """获取所有历史识别记录列表"""
    records = list_records()
    summary = [
        {
            "id": r["id"],
            "filename": r.get("filename"),
            "duration": r.get("duration"),
            "speaker_list": r.get("speaker_list", []),
            "created": r.get("created"),
            "model": r.get("model"),
            "text_preview": r.get("text", "")[:100],
        }
        for r in records
    ]
    return JSONResponse(summary)


@app.get("/api/records/{record_id}")
async def get_record_api(record_id: str, _: bool = Depends(require_token)):
    """获取单条识别记录的完整详情"""
    rec, _ = find_record(record_id)
    if not rec:
        raise HTTPException(404, "Record not found")
    return JSONResponse(rec)


@app.get("/api/records/{record_id}/audio")
async def download_audio_api(record_id: str, _: bool = Depends(require_token)):
    """下载原始音频文件"""
    rec, _ = find_record(record_id)
    if not rec or not rec.get("audio_file"):
        raise HTTPException(404, "Record not found")
    path = os.path.join(AUDIO_DIR, rec["audio_file"])
    if not os.path.exists(path):
        raise HTTPException(404, "Audio file missing")
    return FileResponse(path, filename=rec.get("filename") or rec["audio_file"])


@app.get("/api/records/{record_id}/result")
async def download_result_api(record_id: str, _: bool = Depends(require_token)):
    """下载识别结果 JSON"""
    rec, path = find_record(record_id)
    if not rec:
        raise HTTPException(404, "Record not found")
    base = os.path.splitext(rec.get("filename") or record_id)[0]
    return JSONResponse(rec)


def record_to_md(rec):
    """将识别记录渲染为 Markdown，包含说话人分离标签与时间戳。"""
    name = rec.get("filename") or rec.get("id", "")
    created = rec.get("created", "")
    model = rec.get("model", "")
    duration = rec.get("duration")
    text = rec.get("text", "")

    segments = rec.get("segments") or []
    speakers = [s for s in sorted(set((seg.get("speaker") or "未知") for seg in segments if seg.get("text"))) if s]

    lines = []
    lines.append(f"# {name}")
    lines.append("")
    if created:
        lines.append(f"- **识别时间**: {created}")
    lines.append(f"- **模型**: {model}")
    lines.append(f"- **时长**: {duration}s" if duration is not None else "")
    if speakers:
        lines.append(f"- **说话人**: {', '.join(speakers)}")
    lines.append("")

    lines.append("## 说话人分离结果")
    lines.append("")
    if not segments:
        lines.append("（无分段信息）")
        lines.append("")
        lines.append("> 完整文本：")
        lines.append(">")
        for para in text.split("\n"):
            lines.append(f"> {para}")
    else:
        lines.append("| 说话人 | 开始 (s) | 结束 (s) | 文本 |")
        lines.append("|--------|---------|---------|------|")
        for seg in segments:
            spk = seg.get("speaker") or "未知"
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            t = (seg.get("text") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {spk} | {start} | {end} | {t} |")
    lines.append("")

    # 按时间顺序的分段视图
    lines.append("### 分段详情")
    lines.append("")
    for seg in segments:
        spk = seg.get("speaker") or "未知"
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        t = (seg.get("text") or "").strip()
        lines.append(f"- **[{spk}]** `{start}s - {end}s`: {t}")
    lines.append("")

    # 完整文本
    lines.append("## 完整文本")
    lines.append("")
    for para in text.split("\n"):
        lines.append(para.strip())
    lines.append("")

    return "\n".join(lines)


@app.get("/api/records/{record_id}/md")
async def export_md_api(record_id: str, _: bool = Depends(require_token)):
    """导出 Markdown（含说话人分离 + 时间戳）"""
    rec, _ = find_record(record_id)
    if not rec:
        raise HTTPException(404, "Record not found")
    base = os.path.splitext(rec.get("filename") or record_id)[0]
    content = record_to_md(rec)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base}.md"'},
    )


@app.delete("/api/records/{record_id}")
async def delete_record_api(record_id: str, _: bool = Depends(require_token)):
    """删除识别记录及原始音频文件"""
    rec, rec_path = find_record(record_id)
    if not rec:
        raise HTTPException(404, "Record not found")

    audio_path = os.path.join(AUDIO_DIR, rec["audio_file"]) if rec.get("audio_file") else None
    result_path = os.path.join(RESULT_DIR, f"{record_id}.json")

    for p in [rec_path, result_path, audio_path]:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                logger.warning(f"Failed to delete {p}: {e}")

    return JSONResponse({"status": "ok", "deleted": record_id})


def main():
    parser = argparse.ArgumentParser(description="FunASR Web Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="sensevoice")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    global DEFAULT_MODEL, DEVICE
    DEVICE = args.device
    DEFAULT_MODEL = args.model

    # 启动即预加载默认模型（含说话人分离 CAM++），避免首次请求冷启动卡顿
    load_model(args.model, with_spk=True)

    logger.info(f"FunASR Web Service starting on http://{args.host}:{args.port}")
    logger.info(f"  Device: {DEVICE} | Model: {args.model} | Workers: {args.workers}")
    logger.info(f"  Data dir: {DATA_DIR}")
    logger.info(f"  Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
