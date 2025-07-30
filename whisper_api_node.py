import os
import sys
import numpy as np
import torch
import random
import tempfile
import soundfile as sf
import time
import re
import json
from faster_whisper import WhisperModel

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class WhisperApiNode:
    """Wrapper for official IndexTTS implementation"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wav_path": ("STRING", {"default": "reference.wav", "description": "音频路径"}),
                "model_path": ("STRING", {"default": "Belle-faster-whisper-large-v3-zh-punct-int8", "description": "模型路径"}),
            },
            "optional": {
                "output_srt_file": ("STRING", {"default": "", "description": "输出字幕文件路径"}),
                "reference_text": ("STRING", {"default": "", "description": "字幕原始参考文本，可以防止字幕字词错误。"}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("ocr",)
    FUNCTION = "generate"
    CATEGORY = "audio"
    
    def __init__(self):
        # 根路径
        self.models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "whisper")
        self.ocr_model = None
    
    def generate(self, wav_path, model_path, output_srt_file, reference_text):
        if self.ocr_model is None:
            model_dir = os.path.join(self.models_root, model_path)
            self.ocr_model = WhisperModel(model_dir, device="cuda", compute_type="int8")
        segments, _ = self.ocr_model.transcribe(wav_path, beam_size=5, language="zh", word_timestamps=True, task="transcribe", vad_filter=True)
        print(segments)
        # 初始化文本列表和单词SRT列表
        # 定义分割句子的正则表达式模式
        SENTENCE_SPLIT_PATTERN = r'[，。！？；.!?;……]'

        # 遍历音频片段生成时间戳（不存储texts和words_srt完整结构）
        words_info = []
        texts = []
        for segment in segments:
            texts.append(segment.text)
            if output_srt_file != "":
                for word in segment.words:
                    words_info.append({
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "word": word.word
                    })
        if output_srt_file != "" and reference_text == "":
            reference_text = ".".join(texts)
        # 生成字幕文件
        if output_srt_file != "" and len(words_info) > 0:
            # 按标点分割参考文本
            text_content = [sentence.strip() for sentence in re.split(SENTENCE_SPLIT_PATTERN, reference_text) if sentence.strip()]
            # 计算总时长，修复变量名错误
            try:
                total_duration = words_info[-1]['end']
            except (IndexError, KeyError):
                print("无法获取有效的时间戳信息")
                return

            # 计算每个字符对应的时间
            total_chars = sum(len(sentence) for sentence in text_content)
            if total_chars == 0:
                print("参考文本字符数为0，无法计算时间分配")
                return
            time_per_char = total_duration / total_chars

            # 生成SRT格式的字幕文本
            srt_text = ""
            current_time = 0
            for index, sentence in enumerate(text_content, start=1):
                start_time = current_time
                end_time = start_time + len(sentence) * time_per_char

                srt_text += f"{index}\n"
                srt_text += f"{self._format_time(start_time)} --> {self._format_time(end_time)}\n"
                srt_text += f"{sentence}\n\n"

                current_time = end_time
            # 写入SRT文件
            with open(output_srt_file, 'w', encoding='utf-8') as srt_file:
                srt_file.write(srt_text)

        # 返回 ocr
        return texts
        
    def _format_time(self, seconds):
        """
        静态方法，用于格式化时间为SRT格式。

        参数:
        seconds (float): 需要格式化的秒数

        返回:
        str: 格式化后的时间字符串，格式为"HH:MM:SS,mmm"
        """
        hours = int(seconds // 3600)
        remaining_seconds = seconds % 3600
        minutes = int(remaining_seconds // 60)
        seconds_part = remaining_seconds % 60
        seconds = int(seconds_part)
        milliseconds = int((seconds_part - seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"