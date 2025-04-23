
# ComfyUI-Index-TTS

使用IndexTTS模型在ComfyUI中实现高质量文本到语音转换的自定义节点。支持中文和英文文本，可以基于参考音频复刻声音特征。

![示例截图1](https://github.com/user-attachments/assets/41960425-f739-4496-9520-8f9cae34ff51)
![示例截图2](https://github.com/user-attachments/assets/1ff0d1d0-7a04-4d91-9d53-cd119250ed67)

## 废话两句

- 生成的很快，真的很快！而且竟然也很像！！！
- 效果很好，感谢小破站的开源哈哈哈哈哈
- 附赠道友B站的传送阵[demo](https://huggingface.co/spaces/IndexTeam/IndexTTS)

## 演示案例

以下是一些实际使用效果演示：

| 参考音频 | 输入文本 | 推理结果 |
|---------|---------|---------|
| <video src="https://github.com/user-attachments/assets/5e8cb570-242f-4a16-8472-8a64a23183fb"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/d8b89db3-5cf5-406f-b930-fa75d13ff0bd"></video> |
| <video src="https://github.com/user-attachments/assets/8e774223-e0f7-410b-ae4e-e46215e47e96"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/6e3e63ed-2d3d-4d5a-bc2e-b42530748fa0"></video> |

- 长文本测试：

<video src="https://github.com/user-attachments/assets/6bfa35dc-1a30-4da0-a4dc-ac3def25452b"></video>


## 更新日志

### 2025-04-23

![微信截图_20250423175608](https://github.com/user-attachments/assets/f2b15d8a-3453-4c88-b609-167b372aab74)


- 新增 **Audio Cleaner** 节点，用于处理TTS输出音频中的混响和杂音问题
  - 该节点可以连接在 Index TTS 节点之后，优化生成音频的质量
  - 主要功能：去除混响、降噪、频率滤波和音频归一化
  - 适用于处理有杂音或混响问题的TTS输出

- 修复了对于transformers版本强依赖的问题

#### Audio Cleaner 参数说明

**必需参数**：
- **audio**: 输入音频（通常为 Index TTS 节点的输出）
- **denoise_strength**: 降噪强度（0.1-1.0，默认0.5）
  - 值越大，降噪效果越强，但可能影响语音自然度
- **dereverb_strength**: 去混响强度（0.0-1.0，默认0.7）
  - 值越大，去混响效果越强，适合处理在回声环境下录制的参考音频

**可选参数**：
- **high_pass_freq**: 高通滤波器频率（20-500Hz，默认100Hz）
  - 用于过滤低频噪音，如环境嗡嗡声
- **low_pass_freq**: 低通滤波器频率（1000-16000Hz，默认8000Hz）
  - 用于过滤高频噪音
- **normalize**: 是否归一化音频（"true"或"false"，默认"true"）
  - 开启可使音量更均衡

#### 使用建议

- 对于有明显混响的音频，将 `dereverb_strength` 设置为 0.7-0.9
- 对于有背景噪音的音频，将 `denoise_strength` 设置为 0.5-0.8
- 如果处理后音频听起来不自然，尝试减小 `dereverb_strength` 和 `denoise_strength`
- 高通和低通滤波器可以微调以获得最佳人声效果


## 功能特点

- 支持中文和英文文本合成
- 基于参考音频复刻声音特征（变声功能）
- 支持调节语速
- 多种音频合成参数控制
- Windows兼容（无需额外依赖）

## 安装

### 安装节点

1. 将此代码库克隆或下载到ComfyUI的`custom_nodes`目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
   ```

2. 安装依赖：

   ```bash
   cd ComfyUI-Index-TTS
   pip install -r requirements.txt
   ```

### 下载模型

1. 从[Hugging Face](https://huggingface.co/IndexTeam/Index-TTS/tree/main)或者[魔搭](https://modelscope.cn/models/IndexTeam/Index-TTS)下载IndexTTS模型文件
2. 将模型文件放置在`ComfyUI/models/Index-TTS`目录中（如果目录不存在，请创建）
3. 模型文件夹结构：
   
   ```
   ComfyUI/models/Index-TTS/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```
   
   确保所有文件都已完整下载，特别是较大的模型文件如`bigvgan_discriminator.pth`(1.6GB)和`gpt.pth`(696MB)。

## 使用方法

1. 在ComfyUI中，找到并添加`Index TTS`节点
2. 连接参考音频输入（AUDIO类型）
3. 输入要转换为语音的文本
4. 调整参数（语言、语速等）
5. 运行工作流获取生成的语音输出

### 示例工作流

项目包含一个基础工作流示例，位于`workflow/workflow.json`，您可以在ComfyUI中通过导入此文件来快速开始使用。

## 参数说明

### 必需参数

- **text**: 要转换为语音的文本（支持中英文）
- **reference_audio**: 参考音频，模型会复刻其声音特征
- **language**: 文本语言选择，可选项：
  - `auto`: 自动检测语言（默认）
  - `zh`: 强制使用中文模式
  - `en`: 强制使用英文模式
- **speed**: 语速因子（0.5~2.0，默认1.0）

### 可选参数

以下参数适用于高级用户，用于调整语音生成质量和特性：

- **temperature** (默认1.0): 控制生成随机性，较高的值增加多样性但可能降低稳定性
- **top_p** (默认0.8): 采样时考虑的概率质量，降低可获得更准确但可能不够自然的发音
- **top_k** (默认30): 采样时考虑的候选项数量
- **repetition_penalty** (默认10.0): 重复内容的惩罚系数
- **length_penalty** (默认0.0): 生成内容长度的调节因子
- **num_beams** (默认3): 束搜索的宽度，增加可提高质量但降低速度
- **max_mel_tokens** (默认600): 最大音频token数量
- **sentence_split** (默认auto): 句子拆分方式

## 音色优化建议

要提高音色相似度：

- 使用高质量的参考音频（清晰、无噪音）
- 尝试调整`temperature`参数（0.7-0.9范围内效果较好）
- 增加`repetition_penalty`（10.0-12.0）可以提高音色一致性
- 对于长文本，确保`max_mel_tokens`足够大

## 故障排除


- 如果出现“模型加载失败”，检查模型文件是否完整且放置在正确目录
- 对于Windows用户，无需额外安装特殊依赖，节点已优化
- 如果显示CUDA错误，尝试重启ComfyUI或减少`num_beams`值


## 鸣谢

- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
- 感谢ComfyUI社区的支持
- 感谢使用！

**注**：RTF (Real-Time Factor) 是实时因子，值越小表示生成速度越快。例如RTF=0.3表示生成10秒的音频只需要3秒。

## 许可证

请参考原始IndexTTS项目许可证。
