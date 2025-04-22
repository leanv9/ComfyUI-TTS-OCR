import os
import sys
import numpy as np
import torch
import tempfile
import soundfile as sf
import time
from pathlib import Path

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入TTS模型
from .tts_models import IndexTTSModel


# IndexTTS节点
class IndexTTSNode:
    """
    ComfyUI的IndexTTS节点，用于文本到语音合成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，这是一段测试文本。"}),
                "reference_audio": ("AUDIO", ),
                "language": (["auto", "zh", "en"], {"default": "auto"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 600, "min": 100, "max": 1500, "step": 50}),
                "sentence_split": (["auto", "manual"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"
    
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "Index-TTS")
        self.tts_model = None
        print(f"[IndexTTS] 初始化节点，模型目录设置为: {self.model_dir}")
        
        # 检查模型目录是否存在
        if not os.path.exists(self.model_dir):
            print(f"[IndexTTS] 警告: 模型目录不存在: {self.model_dir}")
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"[IndexTTS] 已创建模型目录: {self.model_dir}")
        else:
            # 检查模型文件
            model_files = os.listdir(self.model_dir)
            print(f"[IndexTTS] 模型目录内容: {model_files}")
    
    def _init_model(self):
        """初始化TTS模型（延迟加载）"""
        if self.tts_model is None:
            print(f"[IndexTTS] 开始加载模型...")
            # 检查必要的模型文件
            required_files = ["gpt.pth", "config.yaml"]
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    file_size = os.path.getsize(file_path) / (1024*1024)  # 转换为MB
                    print(f"[IndexTTS] 找到模型文件: {file} ({file_size:.2f}MB)")
            
            if missing_files:
                error_msg = f"缺少必要的模型文件: {', '.join(missing_files)}"
                print(f"[IndexTTS] 错误: {error_msg}")
                raise FileNotFoundError(error_msg)
                
            try:
                # 记录开始加载时间
                start_time = time.time()
                
                # 使用tts_models.py中的IndexTTSModel实现
                self.tts_model = IndexTTSModel(model_dir=self.model_dir)
                
                # 记录加载完成时间
                load_time = time.time() - start_time
                print(f"[IndexTTS] 模型已成功加载，耗时: {load_time:.2f}秒")
                
                # 输出模型基本信息
                if hasattr(self.tts_model, 'config'):
                    print(f"[IndexTTS] 模型配置:")  
                    for key, value in vars(self.tts_model.config).items():
                        if not key.startswith('_') and not callable(value):
                            print(f"[IndexTTS]   - {key}: {value}")
                            
                # 检查模型是否有必要的组件
                components = [attr for attr in dir(self.tts_model) if not attr.startswith('_') and not callable(getattr(self.tts_model, attr))]
                print(f"[IndexTTS] 模型组件: {components}")
                
            except Exception as e:
                import traceback
                print(f"[IndexTTS] 初始化模型失败: {e}")
                print(f"[IndexTTS] 错误详情:")
                traceback.print_exc()
                raise RuntimeError(f"初始化IndexTTS模型失败: {e}")
    
    def generate_speech(self, text, reference_audio, language="auto", speed=1.0, temperature=1.0, top_p=0.8, top_k=30, repetition_penalty=10.0, length_penalty=0.0, num_beams=3, max_mel_tokens=600, sentence_split="auto"):
        """
        生成语音的主函数
        
        参数:
            text: 要转换为语音的文本
            reference_audio: 参考音频元组 (audio_data, sample_rate)
            language: 文本语言 (auto, zh, en)
            speed: 语速因子，1.0为正常语速
            
        返回:
            audio: 生成的音频元组 (audio_data, sample_rate)
        """
        try:
            # 延迟加载模型
            if self.tts_model is None:
                self._init_model()
            
            # 处理ComfyUI的音频格式
            processed_audio = None
            
            print(f"[IndexTTS] 接收到参考音频，类型: {type(reference_audio)}")
            
            # 如果是ComfyUI标准格式
            if isinstance(reference_audio, dict) and "waveform" in reference_audio and "sample_rate" in reference_audio:
                waveform = reference_audio["waveform"]
                sample_rate = reference_audio["sample_rate"]
                
                print(f"[IndexTTS] 参考音频格式: ComfyUI字典格式, sample_rate={sample_rate}")
                print(f"[IndexTTS] waveform类型: {type(waveform)}, 形状: {waveform.shape if hasattr(waveform, 'shape') else '未知'}")
                
                # 如果waveform是torch.Tensor，转换为numpy
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                    print(f"[IndexTTS] 已将waveform从tensor转换为numpy, 形状: {waveform.shape if hasattr(waveform, 'shape') else '未知'}")
                
                processed_audio = (waveform, sample_rate)
                
            # 如果已经是元组格式
            elif isinstance(reference_audio, tuple) and len(reference_audio) == 2:
                audio_data, sample_rate = reference_audio
                processed_audio = reference_audio
                print(f"[IndexTTS] 参考音频格式: 元组格式, sample_rate={sample_rate}")
                print(f"[IndexTTS] audio_data类型: {type(audio_data)}, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else '未知'}")
            
            # 如果都不是，报错
            if processed_audio is None:
                print(f"[IndexTTS] 错误: 参考音频格式不正确: {type(reference_audio)}")
                if isinstance(reference_audio, dict):
                    print(f"[IndexTTS] 参考音频字典包含键: {reference_audio.keys()}")
                raise ValueError("参考音频格式不支持，应为 AUDIO 类型")
            
            # 创建临时输出文件
            temp_dir = tempfile.gettempdir()
            temp_output = os.path.join(temp_dir, f"tts_output_{int(time.time())}.wav")
            
            print(f"[IndexTTS] 开始生成语音，文本长度: {len(text)}，语言: {language}，语速: {speed}")
            print(f"[IndexTTS] 文本内容: '{text[:100]}{'...' if len(text) > 100 else ''}'")  # 只打印部分文本
            
            # 记录推理开始时间
            infer_start_time = time.time()
            
            # 调用TTS模型生成语音
            try:
                # 简化调用，只使用基本参数
                # 因为我们的wrapper需要要兼容原始模型接口
                result = self.tts_model.infer(
                    reference_audio=processed_audio, 
                    text=text, 
                    output_path=None,  # 不保存文件，直接返回数据
                    language=language,
                    speed=speed
                )
            except Exception as e:
                print(f"[IndexTTS] 调用模型失败: {e}")
                raise
            
            # 记录推理完成时间
            infer_time = time.time() - infer_start_time
            print(f"[IndexTTS] 语音生成完成，耗时: {infer_time:.2f}秒")
            
            # 处理返回结果
            print(f"[IndexTTS] 模型返回结果类型: {type(result)}")
            
            if isinstance(result, tuple) and len(result) == 2:
                # 返回格式: (sample_rate, audio_data)
                sample_rate, audio_data = result
                print(f"[IndexTTS] 生成的音频样本率: {sample_rate}Hz")
                print(f"[IndexTTS] 生成的音频数据类型: {type(audio_data)}")
                print(f"[IndexTTS] 生成的音频形状: {audio_data.shape if hasattr(audio_data, 'shape') else '未知'}")
                
                # 计算音频长度（秒）
                if hasattr(audio_data, 'shape'):
                    audio_duration = audio_data.shape[-1] / sample_rate
                    print(f"[IndexTTS] 生成的音频长度: {audio_duration:.2f}秒")
                
                # 转换为ComfyUI期望的格式
                # 如果是numpy数组，转换为torch tensor
                if isinstance(audio_data, np.ndarray):
                    print(f"[IndexTTS] 将numpy数组转换为torch tensor")
                    audio_data = torch.tensor(audio_data, dtype=torch.float32)
                    
                print(f"[IndexTTS] 转换前的张量维度: {audio_data.dim()}")
                    
                # 确保音频数据是3D张量 [batch, channels, samples]
                if audio_data.dim() == 1:
                    # [samples] -> [1, 1, samples]
                    audio_data = audio_data.unsqueeze(0).unsqueeze(0)
                    print(f"[IndexTTS] 1D张量调整为3D张量: [1, 1, {audio_data.shape[-1]}]")
                elif audio_data.dim() == 2:
                    # [batch, samples] -> [batch, 1, samples]
                    audio_data = audio_data.unsqueeze(1)
                    print(f"[IndexTTS] 2D张量调整为3D张量: [{audio_data.shape[0]}, 1, {audio_data.shape[-1]}]")
                    
                print(f"[IndexTTS] 最终张量形状: {audio_data.shape}")
                    
                # 返回字典格式，符合ComfyUI音频节点期望
                audio_dict = {
                    "waveform": audio_data,
                    "sample_rate": sample_rate
                }
                
                return (audio_dict,)
            else:
                print(f"错误: 意外的返回格式: {type(result)}")
                raise ValueError(f"TTS模型返回了意外的格式: {type(result)}")
                
        except Exception as e:
            import traceback
            print(f"[IndexTTS] 生成语音失败: {e}")
            print(f"[IndexTTS] 错误详情:")
            traceback.print_exc()
            
            # 生成一个简单的错误提示音频
            sample_rate = 24000
            duration = 1.0  # 1秒
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz警告音
            print(f"[IndexTTS] 生成警告音频作为错误处理")
            
            # 转换为ComfyUI音频格式
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            audio_dict = {
                "waveform": signal_tensor,
                "sample_rate": sample_rate
            }
            
            return (audio_dict,)
