import gradio as gr
import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

CANDIDATES = [
    "a photo of a person",
    "a landscape photo",
    "a futuristic scene",
    "a cute cat",
    "a dog",
    "a building",
    "a product photo",
    "a cinematic scene",
    "a high quality image",
    "a fashion photo",
    "a digital art anime style",
    "a hyper realistic render",
]

def image_to_prompt(img):
    """Gradio callback: infer prompt from an uploaded image (numpy array)."""
    if img is None:
        return "No image uploaded", ""

    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(CANDIDATES).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        scores = (image_features @ text_features.T).softmax(dim=-1)
        idx = scores.argmax().item()

    return CANDIDATES[idx], f"{scores[0][idx].item():.4f}"

def build_demo():
    with gr.Blocks(title="Image-to-Prompt WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Image-to-Prompt WebUI
            上传图片，自动生成 AI 提示词（Image → Prompt）  
            Upload an image and get an AI-generated prompt.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="numpy",
                    label="上传图片 / Upload image"
                )
                analyze_btn = gr.Button("开始分析 / Analyze", variant="primary")
            with gr.Column(scale=1):
                prompt_output = gr.Textbox(
                    label="提示词 / Prompt",
                    lines=2
                )
                score_output = gr.Textbox(
                    label="置信度 / Score",
                    lines=1
                )

        analyze_btn.click(
            fn=image_to_prompt,
            inputs=[image_input],
            outputs=[prompt_output, score_output],
        )

        gr.Markdown(
            """
            ---
            提示：  
            - 建议上传清晰度较高的图片  
            - 可用于 AI 绘画反向提示词分析  
            - 如需批量处理，可直接调用 `app.py` 中的函数
            """
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
