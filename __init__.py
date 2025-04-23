""" 
@title: IndexTTS for ComfyUI
@author: ComfyUI-Index-TTS
@description: ComfyUI接口的工业级零样本文本到语音合成系统
"""

import os
import sys

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入节点定义
from .nodes import IndexTTSNode
from .audio_enhancement import AudioCleanupNode

# 注册ComfyUI节点
NODE_CLASS_MAPPINGS = {
    "IndexTTSNode": IndexTTSNode,
    "AudioCleanupNode": AudioCleanupNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTSNode": "Index TTS",
    "AudioCleanupNode": "Audio Cleaner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
