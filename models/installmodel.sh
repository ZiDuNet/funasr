#!/bin/bash

echo "=============================================="
echo "        FunASR 模型下载（已存在自动跳过）"
echo "=============================================="
echo

# 进入脚本所在目录
cd "$(dirname "$0")" || exit 1

# 安装 modelscope
echo "[1/9] 检查并安装 modelscope..."
pip install modelscope -q
echo "安装完成！"
echo

# ========== 下载函数：存在则跳过 ==========
download() {
    local model_name="$1"
    local save_dir="$2"
    
    if [ -d "$save_dir" ] && [ "$(ls -A "$save_dir")" ]; then
        echo "✅ 已存在，跳过：$save_dir"
        return
    fi
    
    echo "🔽 正在下载：$model_name"
    modelscope download --model "$model_name" --local_dir "$save_dir"
}

# ========== 开始下载 ==========

echo "[2/9] 下载 Fun-ASR-Nano"
download "FunAudioLLM/Fun-ASR-Nano-2512" "./Fun-ASR-Nano"

echo -e "\n[3/9] 下载 SenseVoiceSmall"
download "iic/SenseVoiceSmall" "./SenseVoiceSmall"

echo -e "\n[4/9] 下载 Paraformer-zh"
download "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch" "./Paraformer-zh"

echo -e "\n[5/9] 下载 Paraformer-zh-streaming"
download "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online" "./Paraformer-zh-streaming"

echo -e "\n[6/9] 下载 ct-punc"
download "damo/punc_ct-transformer_cn-en-common-vocab471067-large" "./ct-punc"

echo -e "\n[7/9] 下载 fsmn-vad"
download "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" "./fsmn-vad"

echo -e "\n[8/9] 下载 cam++"
download "iic/speech_campplus_sv_zh-cn-16k-common" "./cam++"

echo -e "\n[9/9] 下载 emotion2vec+large"
download "iic/emotion2vec_plus_large" "./emotion2vec-plus-large"

# ========== 自动生成 模型说明.txt ==========
cat > 模型说明.txt << EOF
==============================================
                   模型说明
==============================================

1. Fun-ASR-Nano
   功能：语音识别 + 时间戳
   支持：31 种语言
   参数量：800M

2. SenseVoiceSmall
   功能：语音识别 + 情感识别 + 事件检测
   支持：中/英/日/韩/粤
   参数量：234M

3. Paraformer-zh
   功能：语音识别 + 时间戳
   支持：中/英
   参数量：220M

4. Paraformer-zh-streaming
   功能：流式实时语音识别
   支持：中/英
   参数量：220M

5. ct-punc
   功能：标点符号恢复
   支持：中/英
   参数量：290M

6. fsmn-vad
   功能：语音活动检测（判断有没有人说话）
   支持：中/英
   参数量：0.4M

7. cam++
   功能：说话人分离/说话人确认（区分是谁在说话）
   参数量：7.2M

8. emotion2vec-plus-large
   功能：情感识别（开心/生气/悲伤/正常等）
   参数量：300M

==============================================
EOF

echo -e "\n📄 已生成 模型说明.txt"
echo -e "\n=============================================="
echo "               ✅ 所有模型处理完成！"
echo "=============================================="