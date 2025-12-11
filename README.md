![Image to Prompt Banner](./assets/banner.svg)

<p align="center">
  <a href="https://github.com/ggcs9210/image-to-prompt/stargazers">
    <img src="https://img.shields.io/github/stars/ggcs9210/image-to-prompt?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/ggcs9210/image-to-prompt/issues">
    <img src="https://img.shields.io/github/issues/ggcs9210/image-to-prompt" alt="GitHub issues">
  </a>
  <img src="https://img.shields.io/github/license/ggcs9210/image-to-prompt" alt="License">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/model-CLIP%20ViT--B%2F32-green" alt="Model">
  <img src="https://img.shields.io/badge/UI-Gradio-lightgrey" alt="Gradio">
</p>

---

# Image-to-Prompt WebUI  
先进的图片反推提示词 AI 工具（支持 Web UI | 基于 CLIP | 开源可商用）

Image-to-Prompt 是一个开源的 AI 工具，它能够对任意图片进行理解分析，并自动生成与图片语义最接近的提示词（Prompt）。该项目基于 **OpenAI CLIP ViT-B/32** 模型构建，并配备友好的 **Gradio Web UI**，可用于：

- AI 绘画提示词反推（Stable Diffusion / Midjourney / Flux 等）  
- 图片语义分析与检索  
- 内容理解与标签生成  
- 多模态实验与教学演示  

> This repository provides an AI-powered **image-to-prompt reverse engineering tool**, built on **OpenAI CLIP** and exposed via a clean **Gradio Web UI**.

---

## 🔍 功能亮点 Features

- 🧠 **自动反推提示词**：上传图片，即可获取最相关的语义提示词  
- 🖥 **可视化 Web UI**：无需命令行，小白也能直接用浏览器操作  
- 🧩 **模块化设计**：核心逻辑与 UI 解耦，便于二次开发与集成  
- ⚙ **支持 CPU / GPU**：在本地环境或云服务器均可运行  
- 🧪 **适合集成到 AI 工作流**：可用作提示词生成模块、数据标注辅助工具  

---

## 📦 项目结构 Project Structure

```bash
image-to-prompt/
├── app.py                 # CLI/脚本模式：从文件路径反推提示词
├── webui.py               # Web UI 主入口（Gradio）
├── spaces_app.py          # HuggingFace Spaces 版本入口（可选）
├── requirements.txt       # Python 依赖列表
├── README.md              # 主 README（当前文件，中文 + English）
├── README_EN.md           # 英文独立版本
├── LICENSE                # MIT 开源协议
├── assets/
│   ├── banner.svg         # GitHub 项目 Banner
│   └── demo.gif           # 占位操作演示动图
└── models/
    └── clip_reverse.py    # 模型扩展逻辑占位文件
```

---

## 🚀 快速开始 Quick Start

### 1. 克隆仓库 Clone

```bash
git clone https://github.com/ggcs9210/image-to-prompt.git
cd image-to-prompt
```

### 2. 安装依赖 Install dependencies

```bash
pip install -r requirements.txt
```

> 建议使用 Python 3.8+ 和虚拟环境（virtualenv / conda）。

### 3. 启动 Web UI（本地）Run Web UI locally

```bash
python webui.py
```

默认在浏览器访问：

```text
http://127.0.0.1:7860
```

如果你想从其他设备访问（同一局域网内），可以把 `webui.py` 中的：

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

保持不变，然后在其他设备用：

```text
http://<你的电脑IP>:7860
```

访问。

### 4. 使用命令行模式（可选）CLI usage

```bash
python app.py
```

或在代码中：

```python
from app import image_to_prompt

prompt, score = image_to_prompt("test.jpg")
print(prompt, score)
```

---

## 🧠 模型与技术原理 Model & Technical Details

本项目使用 **CLIP (Contrastive Language–Image Pre-training)**：

1. 利用 CLIP 对输入图像编码，得到图像特征向量  
2. 准备一组候选提示词（英文短句），并使用 CLIP 编码为文本特征向量  
3. 计算图像特征与每个文本特征的相似度（点积 / 余弦相似度）  
4. 选择相似度最高的提示词作为输出，并返回置信度 score  

你可以在 `webui.py` 中修改 `CANDIDATES` 列表来自定义：

```python
CANDIDATES = [
    "a cinematic scene",
    "a cute cat",
    "a photo of a person",
    "a digital art anime style",
    "a highly detailed portrait",
    "a landscape photo",
    # ...
]
```

未来可以扩展：

- 使用更大的/中文支持更好的视觉语言模型（BLIP2 / Florence2 / Qwen-VL 等）  
- 使用 LLM 对基础提示词进行“润色”，一键生成 Stable Diffusion / Flux 风格长提示词  

---

## 🌐 HuggingFace Spaces 部署（可选）Deploy on HuggingFace Spaces

你可以很方便地把本项目部署到 [HuggingFace Spaces](https://huggingface.co/spaces)：

1. 在 HuggingFace 上创建一个新 Space，选择：  
   - 类型：Gradio  
   - Runtime：Python
2. 把本项目中的以下文件上传到 Space：  
   - `spaces_app.py`（或重命名为 `app.py`）  
   - `requirements.txt`  
3. 确保 Space 中的 `app.py`（或主入口）包含：

```python
# spaces_app.py (示例)
import gradio as gr
from webui import image_to_prompt, CANDIDATES, model, preprocess, device

def launch_spaces():
    with gr.Blocks(title="Image to Prompt WebUI") as demo:
        gr.Markdown("# Image-to-Prompt WebUI\nHuggingFace Spaces Demo")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload image")
            with gr.Column():
                prompt_output = gr.Textbox(label="Prompt")
                score_output = gr.Textbox(label="Score")

        demo_button = gr.Button("Analyze")

        demo_button.click(
            fn=image_to_prompt,
            inputs=[image_input],
            outputs=[prompt_output, score_output],
        )

    return demo

demo = launch_spaces()

if __name__ == "__main__":
    demo.launch()
```

4. 等待 Spaces 自动构建完成后，即可获得一个在线 Demo 链接，放回 GitHub README 中：

```markdown
[在线体验 / Live Demo](https://huggingface.co/spaces/你的空间名)
```

---

## 🎨 Web UI 主题与美化 Theme & UI

当前 Web UI 使用 Gradio Blocks 构建，支持：

- 自定义布局（左图右文）  
- 自定义标题、说明文案  
- 未来可替换主题，例如：

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    ...
```

你也可以：

- 添加历史记录区域  
- 添加“复制提示词”按钮  
- 增加语言选择（中/英）切换  

---

## 🎬 项目演示 Demo

占位演示 GIF：

![Demo](./assets/demo.gif)

> 建议你后续用真实 Web UI 录制一个 GIF 替换此文件，使项目看起来更专业。

推荐录制方式：

- Windows：ScreenToGif  
- macOS：Kap / CleanShot X  
- Linux：Peek  

---

## 🗺 Roadmap

- [ ] 中文提示词反推与多语言支持  
- [ ] 一键生成 Stable Diffusion / Flux 风格长提示词  
- [ ] HuggingFace 在线 Demo 官方版本  
- [ ] Web UI 主题美化与深色模式切换  
- [ ] 批量图片处理与导出  
- [ ] 支持更多视觉语言模型（BLIP2 / Florence2 / Qwen-VL 等）  

---

## 🤝 Contributing

欢迎通过以下方式参与：

- 提交新的候选提示词库  
- 集成新的视觉模型  
- 优化 Web UI 交互与样式  
- 修复 Bug / 改善文档 / 添加多语言支持  

---

## 📄 License

本项目使用 **MIT License**，可在保留版权声明的前提下自由商用与二次分发。
