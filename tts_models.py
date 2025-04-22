import os
import sys
import yaml
import time
import torch
import numpy as np
import tempfile
import soundfile as sf
import traceback
import torchaudio
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入官方IndexTTS实现
from indextts.infer import IndexTTS


class IndexTTSModel:
    """Wrapper for official IndexTTS implementation"""
    
    def __init__(self, model_dir="models/Index-TTS"):
        """
        Initialize IndexTTS model
        
        Args:
            model_dir: Model directory path
        """
        print(f"[IndexTTS-Model] ==== 初始化IndexTTS模型器 =====")
        print(f"[IndexTTS-Model] 输入模型目录: {model_dir}")
        
        # 解析路径
        self.model_dir = model_dir
        self.abs_model_dir = os.path.abspath(model_dir)
        print(f"[IndexTTS-Model] 模型绝对路径: {self.abs_model_dir}")
        
        # 检查设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[IndexTTS-Model] 使用设备: {self.device}")
        if self.device.startswith("cuda"):
            print(f"[IndexTTS-Model] CUDA可用，显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
            print(f"[IndexTTS-Model] 当前GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = None
        self.cfg_path = os.path.join(self.model_dir, "config.yaml")
        print(f"[IndexTTS-Model] 配置路径: {self.cfg_path}")
        
        try:
            # 创建目录
            if not os.path.exists(self.model_dir):
                print(f"[IndexTTS-Model] 模型目录不存在，创建路径: {self.model_dir}")
                os.makedirs(self.model_dir, exist_ok=True)
            
            # 检查配置文件
            if os.path.exists(self.cfg_path):
                print(f"[IndexTTS-Model] 配置文件存在: {self.cfg_path}")
                file_size = os.path.getsize(self.cfg_path) / 1024  # KB
                print(f"[IndexTTS-Model] 配置文件大小: {file_size:.2f}KB")
            else:
                print(f"[IndexTTS-Model] 错误: 配置文件不存在: {self.cfg_path}")
                raise FileNotFoundError(f"配置文件不存在: {self.cfg_path}")
                
            # 检查必要模型文件
            required_files = [
                "bigvgan_generator.pth", 
                "bpe.model", 
                "gpt.pth", 
                "config.yaml"
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"[IndexTTS-Model] 必要模型文件存在: {file} ({file_size:.2f}MB)")
                else:
                    missing_files.append(file)
                    print(f"[IndexTTS-Model] 警告: 模型文件 {file} 不存在，可能会影响模型正常运行")
            
            if missing_files:
                print(f"[IndexTTS-Model] 缺失文件列表: {missing_files}")
                raise FileNotFoundError(f"缺失必要模型文件: {missing_files}")
            
            # 列出模型目录中的所有文件
            print(f"[IndexTTS-Model] 模型目录内容: {os.listdir(self.model_dir)}")
            
            print(f"[IndexTTS-Model] 模型检查完成！")
            
            # 在初始化时就加载模型，而不是延迟加载
            print(f"[IndexTTS-Model] 开始加载模型实例...")
            self._lazy_load_model()
            
        except Exception as e:
            import traceback
            print(f"[IndexTTS-Model] 检查模型文件失败: {e}")
            print(f"[IndexTTS-Model] 错误详情:")
            traceback.print_exc()
            raise
    
    def _lazy_load_model(self):
        """
        Lazy load model, only when needed
        
        Returns:
            bool: Success or not
        """
        if self.model is not None:
            return True
            
        print(f"[IndexTTS-Model] 开始加载模型...")
        start_time = time.time()
        
        try:
            # 确保可以访问到indextts模块
            indextts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indextts')
            if indextts_path not in sys.path and os.path.exists(indextts_path):
                print(f"[IndexTTS-Model] 将{indextts_path}添加到sys.path")
                sys.path.append(indextts_path)
                
            # 确保可以访问到父目录
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_path not in sys.path:
                print(f"[IndexTTS-Model] 将{parent_path}添加到sys.path")
                sys.path.append(parent_path)
            
            # 使用设备
            device = self.device
            is_fp16 = False if device == "cpu" else True
            use_cuda_kernel = device.startswith("cuda")
            
            print(f"[IndexTTS-Model] 创建模型实例, 使用设备: {device}, FP16: {is_fp16}, CUDA Kernel: {use_cuda_kernel}")
            
            # 创建简单的模型调用包装器
            # 定义前处理函数，在调用IndexTTS实例之前修复路径问题
            def patch_model_paths(module_name):
                orig_import = __import__
                
                def custom_import(name, *args, **kwargs):
                    module = orig_import(name, *args, **kwargs)
                    if name == module_name:
                        # 修改IndexTTS.__init__方法以加载正确路径的BPE模型
                        orig_init = module.IndexTTS.__init__
                        
                        def patched_init(self, *args, **kwargs):
                            # 调用原始函数
                            result = orig_init(self, *args, **kwargs)
                            
                            # 修复BPE模型路径
                            # 检查是否包含"checkpoints/"
                            if hasattr(self, 'bpe_path') and 'checkpoints/' in self.bpe_path:
                                orig_path = self.bpe_path
                                # 移除checkpoints/路径
                                corrected_path = os.path.join(self.model_dir, os.path.basename(self.bpe_path))
                                if os.path.exists(corrected_path):
                                    print(f"[IndexTTS-Model] 修复bpe_path路径: {orig_path} -> {corrected_path}")
                                    self.bpe_path = corrected_path
                                    # 重新加载分词器
                                    self.tokenizer = spm.SentencePieceProcessor(model_file=self.bpe_path)
                                    print(f"[IndexTTS-Model] 成功重新加载分词器: {self.bpe_path}")
                            
                            return result
                        
                        # 替换原始的__init__方法
                        module.IndexTTS.__init__ = patched_init
                    return module
                
                return custom_import
            
            # 保存原始导入函数
            original_import = __import__
            
            # 替换导入函数为包装版本
            __builtins__['__import__'] = patch_model_paths('indextts.infer')
            
            try:
                # 现在加载模型
                model = IndexTTS(
                    cfg_path=self.cfg_path, 
                    model_dir=self.model_dir,
                    is_fp16=is_fp16,
                    device=device,
                    use_cuda_kernel=use_cuda_kernel
                )
            finally:
                # 恢复原始导入函数
                __builtins__['__import__'] = original_import
            
            # 检查并直接修复bpe_path
            if hasattr(model, 'bpe_path') and not os.path.exists(model.bpe_path):
                # 尝试直接在模型目录根目录下查找
                base_name = os.path.basename(model.bpe_path)
                corrected_path = os.path.join(self.model_dir, base_name)
                if os.path.exists(corrected_path):
                    print(f"[IndexTTS-Model] 手动修复bpe_path路径: {model.bpe_path} -> {corrected_path}")
                    model.bpe_path = corrected_path
                    # 重新对分词器初始化
                    import sentencepiece as spm
                    model.tokenizer = spm.SentencePieceProcessor(model_file=model.bpe_path)
                    print(f"[IndexTTS-Model] 成功重新加载分词器: {model.bpe_path}")
            
            # 测试模型是否有infer方法
            if not hasattr(model, 'infer'):
                print(f"[IndexTTS-Model] 错误: 模型没有infer方法")
                raise AttributeError("Model does not have 'infer' method")
            
            self.model = model
            load_time = time.time() - start_time
            print(f"[IndexTTS-Model] 模型成功加载，耗时: {load_time:.2f}秒")
            return True
        except Exception as e:
            self.model = None  # 确保在失败时设置为None
            print(f"[IndexTTS-Model] 模型加载失败: {e}")
            print(f"[IndexTTS-Model] 错误详情:")
            traceback.print_exc()
            raise ValueError(f"无法加载 IndexTTS 模型: {e}")
        return False
        
    def infer_fast(self, reference_audio, text, output_path=None, language="auto", speed=1.0, verbose=False, temperature=1.0, top_p=0.8, top_k=30, repetition_penalty=10.0, length_penalty=0.0, num_beams=3, max_mel_tokens=600, bucket_enable=True):
        """
        Fast inference method (fallback to regular infer as this is a wrapper)
        
        Args:
            reference_audio: Reference audio path or audio data tuple (audio_data, sample_rate)
            text: Text to synthesize
            output_path: Output file path (optional)
            language: Text language (auto, zh, en)
            speed: Speech speed factor
            verbose: Whether to print detailed logs
            temperature: Model temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for length
            num_beams: Number of beams for beam search
            max_mel_tokens: Maximum number of mel tokens
            bucket_enable: Whether to enable sentence bucketing
            
        Returns:
            If output_path is provided: output_path
            Otherwise: (sample_rate, audio_data)
        """
        # 由于我们是wrapper，这里实际上直接调用普通推理方法，不实现快速推理
        print(f"[IndexTTS-Model] 调用快速推理模式 (回退到普通推理)")
        return self.infer(
            reference_audio=reference_audio,
            text=text,
            output_path=output_path,
            language=language,
            speed=speed,
            verbose=verbose,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_beams=num_beams,
            max_mel_tokens=max_mel_tokens
        )
            
    def infer(self, reference_audio, text, output_path=None, language="auto", speed=1.0, verbose=False, temperature=1.0, top_p=0.8, top_k=30, repetition_penalty=10.0, length_penalty=0.0, num_beams=3, max_mel_tokens=600):
        """
        Generate speech using reference audio
        
        Args:
            reference_audio: Reference audio path or audio data tuple (audio_data, sample_rate)
            text: Text to synthesize
            output_path: Output file path (optional)
            language: Text language (auto, zh, en)
            speed: Speech speed factor
            verbose: Whether to print detailed logs
            temperature: Model temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for length
            num_beams: Number of beams for beam search
            max_mel_tokens: Maximum number of mel tokens
            
        Returns:
            If output_path is provided: output_path
            Otherwise: (sample_rate, audio_data)
        """
        print(f"[IndexTTS-Model] ==== 开始语音合成 =====")
        print(f"[IndexTTS-Model] 文本长度: {len(text)}, 语言: {language}, 语速: {speed}")
        print(f"[IndexTTS-Model] 文本内容: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"[IndexTTS-Model] 参考音频类型: {type(reference_audio)}")
        print(f"[IndexTTS-Model] 输出路径: {output_path if output_path else '无（将返回音频数据）'}")
        
        # 确保模型已加载
        if self.model is None:
            print(f"[IndexTTS-Model] 模型尚未加载，尝试加载...")
            self._lazy_load_model()
        
        # 记录开始时间
        infer_start_time = time.time()
        
        try:
            # 处理参考音频
            print(f"[IndexTTS-Model] 处理参考音频")
            final_ref_audio = None
            
            if isinstance(reference_audio, str):
                if not os.path.exists(reference_audio):
                    print(f"[IndexTTS-Model] 错误: 参考音频文件不存在: {reference_audio}")
                    raise FileNotFoundError(f"参考音频文件不存在: {reference_audio}")
                print(f"[IndexTTS-Model] 使用文件路径作为参考音频: {reference_audio}")
                final_ref_audio = reference_audio
            else:
                # 假设输入是(audio_data, sample_rate)元组
                try:
                    audio_data, sample_rate = reference_audio
                    print(f"[IndexTTS-Model] 使用音频数据作为参考，采样率: {sample_rate}Hz, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
                    
                    # 使用与原始实现相同的方式准备参考音频
                    # 首先确保数据是torch tensor
                    if isinstance(audio_data, np.ndarray):
                        print(f"[IndexTTS-Model] 将NumPy数组转换为Torch张量")
                        audio_data = torch.from_numpy(audio_data)
                    
                    print(f"[IndexTTS-Model] 原始张量形状: {audio_data.shape}, 维度: {audio_data.dim()}D")
                    
                    # 处理音频数据维度 - 直接将其转换为2D
                    if audio_data.dim() == 3 and audio_data.shape[0] == 1:  # [1, channels, samples]
                        print(f"[IndexTTS-Model] 将 [1, channels, samples] 转换为 [channels, samples]")
                        audio_data = audio_data.squeeze(0)  # 去除批次维度，变成 [channels, samples]
                    
                    # 如果还是3D或更高维度，尝试其他方法
                    if audio_data.dim() > 2:
                        print(f"[IndexTTS-Model] 张量仍然是{audio_data.dim()}D，尝试将其平均为2D")
                        # 如果是 [batch, channels, samples] 格式，选择第一个批次和从多声道取均值
                        if audio_data.dim() == 3:
                            audio_data = audio_data[0]  # 选择第一个批次 -> [channels, samples]
                            if audio_data.shape[0] > 1:  # 多声道
                                audio_data = torch.mean(audio_data, dim=0, keepdim=True)  # 声道平均值 -> [1, samples]
                    
                    # 再次检查维度，确保是2D
                    if audio_data.dim() > 2:
                        print(f"[IndexTTS-Model] 警告: 重试降维后仍然是 {audio_data.dim()}D 张量，尝试直接重塑形状")
                        # 最后手段: 强制将所有数据打平成 [1, -1] 形状
                        audio_data = audio_data.reshape(1, -1)
                    
                    print(f"[IndexTTS-Model] 处理后的张量形状: {audio_data.shape}, 维度: {audio_data.dim()}D")
                    
                    # 确保采样率正确
                    if sample_rate != 24000:
                        print(f"[IndexTTS-Model] 将采样率从 {sample_rate}Hz 转换为 24000Hz")
                        audio_data = torchaudio.transforms.Resample(sample_rate, 24000)(audio_data)
                        sample_rate = 24000
                    
                    # 创建临时文件保存音频数据
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                        print(f"[IndexTTS-Model] 将处理后的音频数据保存为临时文件: {temp_path}")
                        torchaudio.save(temp_path, audio_data, sample_rate)
                        final_ref_audio = temp_path
                        print(f"[IndexTTS-Model] 成功保存临时文件: {temp_path}")
                except Exception as e:
                    print(f"[IndexTTS-Model] 错误: 处理音频数据时出错: {e}")
                    raise ValueError(f"无法处理参考音频数据: {e}")
            
            # 构造输出路径
            final_output_path = output_path
            if final_output_path is None:
                # 创建临时文件作为输出路径
                temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                final_output_path = temp_output.name
                temp_output.close()  # 关闭但不删除
                print(f"[IndexTTS-Model] 未提供输出路径，将使用临时文件: {final_output_path}")
            
            # 这里调用实际的模型推理
            print(f"[IndexTTS-Model] 开始生成语音，文本长度: {len(text)}")
            start_time = time.time()
            
            # 再次检查模型是否加载成功
            if self.model is None:
                print(f"[IndexTTS-Model] 错误: 模型未成功加载，无法执行推理")
                raise ValueError("模型未成功加载，请检查模型文件和错误日志")
            
            # 使用原始IndexTTS模型进行推理
            # 注意：原始IndexTTS模型的infer方法没有verbose、temperature等参数
            # 我们只传递它支持的基本参数，避免错误
            if len(text) > 100:
                print(f"[IndexTTS-Model] 文本较长，使用批量推理模式")
                try:
                    # 直接使用基础参数，避免传递额外参数
                    result = self.model.infer(final_ref_audio, text, final_output_path)
                except Exception as e:
                    print(f"[IndexTTS-Model] 批量推理失败: {e}，回退到普通推理模式")
                    result = self.model.infer(final_ref_audio, text, final_output_path)
            else:
                print(f"[IndexTTS-Model] 文本较短，使用普通推理模式")
                result = self.model.infer(final_ref_audio, text, final_output_path)
            
            end_time = time.time()
            print(f"[IndexTTS-Model] 语音生成完成，耗时: {end_time - start_time:.2f}秒")
            
            # 根据输出类型返回结果
            if output_path:
                print(f"[IndexTTS-Model] 返回保存的文件路径: {output_path}")
                return output_path
            else:
                if isinstance(result, tuple) and len(result) == 2:
                    # 如果result已经是(sample_rate, audio_data)格式
                    sample_rate, audio_data = result
                    print(f"[IndexTTS-Model] 返回内存中的音频数据，采样率: {sample_rate}Hz, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
                    return result
                elif os.path.exists(final_output_path):
                    # 读取临时文件并返回数据
                    audio_data, sample_rate = sf.read(final_output_path)
                    print(f"[IndexTTS-Model] 返回内存中的音频数据，采样率: {sample_rate}Hz, 形状: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
                    # 删除临时文件
                    try:
                        os.unlink(final_output_path)
                    except Exception as e:
                        print(f"[IndexTTS-Model] 警告: 无法删除临时文件: {e}")
                    return (sample_rate, audio_data)
                else:
                    print(f"[IndexTTS-Model] 错误: 输出文件不存在: {final_output_path}")
                    raise FileNotFoundError(f"输出文件不存在: {final_output_path}")
        
        except Exception as e:
            print(f"[IndexTTS-Model] 语音生成失败: {e}")
            print(f"[IndexTTS-Model] 错误详情:")
            traceback.print_exc()
            raise
        finally:
            infer_end_time = time.time()
            print(f"[IndexTTS-Model] 总耗时: {infer_end_time - infer_start_time:.2f}秒")
