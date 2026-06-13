/* WebSocket 客户端封装 */

class FunASRWebSocket {
  constructor(onMessage, onStateChange) {
    this.ws = null;
    this.onMessage = onMessage;
    this.onStateChange = onStateChange;
    this.connected = false;
  }

  connect(options = {}) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = (location.protocol === 'file:' || location.origin === 'null') ? 'localhost:17767' : location.host;
    const token = localStorage.getItem('funasr_token') || '';
    const path = '/api/v1/realtime/transcriptions';
    const wsUrl = token ? `${proto}//${host}${path}?token=${encodeURIComponent(token)}` : `${proto}//${host}${path}`;
    this.ws = new WebSocket(wsUrl, ['binary']);
    this.ws.binaryType = 'arraybuffer';

    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error('WebSocket 连接超时')), 8000);

      this.ws.onopen = () => {
        this.connected = true;
        this.onStateChange?.('connected');
        this.ws.send(JSON.stringify(this.buildSessionConfig(options)));
        window.clearTimeout(timeout);
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.onMessage?.(data);
        } catch (e) {
          console.error('WebSocket 解析错误:', e);
        }
      };

      this.ws.onclose = () => {
        this.connected = false;
        window.clearTimeout(timeout);
        this.onStateChange?.('disconnected');
      };

      this.ws.onerror = (e) => {
        console.error('WebSocket 错误:', e);
        window.clearTimeout(timeout);
        this.onStateChange?.('error');
        reject(new Error('WebSocket 连接失败'));
      };
    });
  }

  buildSessionConfig(options) {
    const hotwords = this.parseHotwords(options.hotwords);
    const speakerGroup = (options.speakerGroup || '').trim();
    const speakerMatch = Boolean(options.speakerMatch && speakerGroup);
    const features = {
      diarization: Boolean(options.diarization || speakerMatch),
      speaker_match: speakerMatch
        ? { enabled: true, group_id: speakerGroup }
        : { enabled: false },
      emotion: Boolean(options.emotion),
      events: Boolean(options.events),
      punctuation: options.punctuation !== false,
      raw: Boolean(options.raw),
    };
    if (speakerGroup) features.speaker_group = speakerGroup;

    const cfg = {
      type: 'session.start',
      mode: options.mode || '2pass',
      audio_fs: 16000,
      wav_format: 'pcm',
      wav_name: 'h5_microphone',
      chunk_size: [0, 10, 5],
      chunk_interval: 10,
      encoder_chunk_look_back: 4,
      decoder_chunk_look_back: 1,
      itn: true,
      features,
      options: {
        language: options.language || 'auto',
      },
      fallback: options.fallback || 'auto',
    };
    if (hotwords) cfg.options.hotwords = hotwords;
    return cfg;
  }

  parseHotwords(value) {
    const text = (value || '').trim();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (e) {
      throw new Error('热词必须是合法 JSON，例如 {"FunASR": 20}');
    }
  }

  sendAudio(pcmData) {
    if (this.ws && this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(pcmData);
    }
  }

  stop() {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ type: 'audio.end' }));
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }
}

// PCM 录音处理器
class PCMRecorder {
  constructor(onData, sampleRate = 16000) {
    this.onData = onData;
    this.sampleRate = sampleRate;
    this.context = null;
    this.stream = null;
    this.processor = null;
    this.mutedOutput = null;
    this.recording = false;
  }

  async start() {
    // 浏览器安全策略：麦克风仅限 HTTPS 或 localhost
    if (!navigator.mediaDevices?.getUserMedia) {
      const isFile = location.protocol === 'file:';
      const isHttp = location.protocol === 'http:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1';
      if (isFile) {
        throw new Error('请通过 http://localhost:17767 访问，不要直接打开 HTML 文件。');
      }
      if (isHttp) {
        throw new Error('浏览器安全策略禁止 HTTP 页面访问麦克风。请使用 http://localhost 或配置 HTTPS 访问。');
      }
      throw new Error('浏览器不支持麦克风访问，请使用最新版 Chrome/Edge。');
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });
    this.context = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: this.sampleRate });
    if (this.context.state === 'suspended') await this.context.resume();
    const source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.mutedOutput = this.context.createGain();
    this.mutedOutput.gain.value = 0;

    this.processor.onaudioprocess = (e) => {
      if (!this.recording) return;
      const data = e.inputBuffer.getChannelData(0);
      const pcm16 = this.resampleToInt16(data, e.inputBuffer.sampleRate || this.context.sampleRate);
      this.onData(pcm16.buffer);
    };

    source.connect(this.processor);
    this.processor.connect(this.mutedOutput);
    this.mutedOutput.connect(this.context.destination);
    this.recording = true;
  }

  stop() {
    this.recording = false;
    if (this.processor) this.processor.disconnect();
    if (this.mutedOutput) this.mutedOutput.disconnect();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.context) this.context.close();
  }

  resampleToInt16(float32, sourceRate) {
    if (!sourceRate || sourceRate === this.sampleRate) return this.float32ToInt16(float32);

    const ratio = sourceRate / this.sampleRate;
    const length = Math.max(1, Math.round(float32.length / ratio));
    const resampled = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      const srcIndex = i * ratio;
      const before = Math.floor(srcIndex);
      const after = Math.min(before + 1, float32.length - 1);
      const weight = srcIndex - before;
      resampled[i] = float32[before] * (1 - weight) + float32[after] * weight;
    }
    return this.float32ToInt16(resampled);
  }

  float32ToInt16(samples) {
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
  }
}
