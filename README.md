# Image-to-Prompt WebUI  
先进的图片反推提示词 AI 工具（支持 Web UI | 基于 CLIP | 开源可商用）

Image-to-Prompt 是一个开源的 AI 工具，它能够对任意图片进行理解分析，并自动生成与图片语义最接近的提示词（Prompt）。该项目基于 OpenAI CLIP 模型构建，并配备友好的 Web UI，可用于 AI 绘画提示词反推、图像语义分析、内容理解等任务。

This project provides an AI-powered **Image-to-Prompt reverse engineering tool**, implemented with **OpenAI CLIP**, and equipped with a clean **Gradio Web UI**. It can analyze an image and generate the most relevant textual prompt concepts automatically.

---

# ✨ 功能亮点 Features

## ✔ 1. 自动反推提示词（Reverse Prompt Engineering）
上传图片后，系统会自动分析图片中的语义内容，生成最可能的提示词。

## ✔ 2. Web UI 可视化界面（无需命令行）
支持拖拽上传图片，自动识别内容，显示提示词与置信度。

## ✔ 3. 基于 CLIP 多模态模型
使用 OpenAI CLIP 提取图像与文本语义匹配分数。

## ✔ 4. 支持二次开发
可用于 Stable Diffusion 反推提示词、标签生成、图像分类等。

---

# 🚀 快速开始 Quick Start

## 安装依赖
```
pip install -r requirements.txt
```

## 启动 Web UI
```
python webui.py
```

访问：
```
http://127.0.0.1:7860
```

---

# 📦 项目结构
```
image-to-prompt/
├── app.py
├── webui.py
├── requirements.txt
├── README.md
├── LICENSE
└── models/
    └── clip_reverse.py
```

---

# 🧠 技术原理
本项目基于 **CLIP ViT-B/32**，通过余弦相似度匹配图像特征与文本特征，从候选提示词列表中选出最相关的提示词。

---

# 🗺 Roadmap
- [ ] 中文提示词反推  
- [ ] HuggingFace 在线 Demo  
- [ ] Web UI 美化  
- [ ] 批量图像处理  
- [ ] 支持 BLIP2 / Florence2  

---

# 📄 License
MIT License
