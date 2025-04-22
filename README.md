# ComfyUI-Index-TTS

使用IndexTTS模型在ComfyUI中实现高质量文本到语音转换的自定义节点。支持中文和英文文本，可以基于参考音频复刻声音特征。

![示例工作流]![微信截图_20250422170940](https://github.com/user-attachments/assets/41960425-f739-4496-9520-8f9cae34ff51)


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
   git clone https://github.com/chenpipi0807/Comfyui-index-tts.git
   ```

2. 安装依赖：

   ```bash
   cd Comfyui-index-tts
   pip install -r requirements.txt
   ```

### 下载模型

1. 从[Hugging Face](https://huggingface.co/IndexTeam/Index-TTS/tree/main)下载IndexTTS模型文件
2. 将模型文件放置在`ComfyUI/models/Index-TTS`目录中（如果目录不存在，请创建）
3. 确保至少包含以下文件：
   - `bigvgan_generator.pth`
   - `bpe.model`
   - `gpt.pth`
   - `config.yaml`

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

- 如果出现"模型加载失败"，检查模型文件是否完整且放置在正确目录
- 对于Windows用户，无需额外安装特殊依赖，节点已优化
- 如果显示CUDA错误，尝试重启ComfyUI或减少`num_beams`值

## 鸣谢

- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
- 感谢ComfyUI社区的支持
- 

> 注：RTF (Real-Time Factor) 是实时因子，值越小表示生成速度越快。例如RTF=0.3表示生成10秒的音频只需要3秒。

## 许可证

请参考原始IndexTTS项目许可证。
