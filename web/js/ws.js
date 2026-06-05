/* WebSocket 客户端封装 */

class FunASRWebSocket {
  constructor(onMessage, onStateChange) {
    this.ws = null;
    this.onMessage = onMessage;
    this.onStateChange = onStateChange;
    this.connected = false;
  }

  connect(mode = '2pass', chunkSize = '5,10,5', chunkInterval = 10, hotwords = '', speakerDiarization = false, emotion = false, events = false) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = (location.protocol === 'file:' || location.origin === 'null') ? 'localhost:17767' : location.host;
    const token = localStorage.getItem('funasr_token') || '';
    const wsUrl = token ? `${proto}//${host}/ws?token=${encodeURIComponent(token)}` : `${proto}//${host}/ws`;
    this.ws = new WebSocket(wsUrl, ['binary']);

    this.ws.onopen = () => {
      this.connected = true;
      this.onStateChange?.('connected');
      // 发送初始配置
      const cfg = {
        mode,
        chunk_size: chunkSize.split(',').map(Number),
        chunk_interval: chunkInterval,
        wav_name: 'h5_microphone',
        is_speaking: true,
        wav_format: 'pcm',
        audio_fs: 16000,
        itn: true,
        speaker_diarization: speakerDiarization,
        emotion: emotion,
        events: events,
      };
      if (hotwords) cfg.hotwords = hotwords;
      this.ws.send(JSON.stringify(cfg));
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
      this.onStateChange?.('disconnected');
    };

    this.ws.onerror = (e) => {
      console.error('WebSocket 错误:', e);
      this.onStateChange?.('error');
    };
  }

  sendAudio(pcmData) {
    if (this.ws && this.connected) {
      this.ws.send(pcmData);
    }
  }

  stop() {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ is_speaking: false }));
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
      audio: { sampleRate: this.sampleRate, channelCount: 1, echoCancellation: true }
    });
    this.context = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: this.sampleRate });
    const source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.recording) return;
      const data = e.inputBuffer.getChannelData(0);
      const pcm16 = this.float32ToInt16(data);
      this.onData(pcm16.buffer);
    };

    source.connect(this.processor);
    this.processor.connect(this.context.destination);
    this.recording = true;
  }

  stop() {
    this.recording = false;
    if (this.processor) this.processor.disconnect();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.context) this.context.close();
  }

  float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
  }
}
