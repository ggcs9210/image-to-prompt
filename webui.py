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
    if img is None:
        return "No image uploaded", ""

    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(CANDIDATES).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        scores = (image_features @ text_features.T).softmax(dim=-1)
        idx = scores.argmax().item()

    return CANDIDATES[idx], float(scores[0][idx].item())

def launch_webui():
    with gr.Blocks(title="Image to Prompt") as demo:
        gr.Markdown("# Image to Prompt
上传图片即可自动生成提示词")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="上传图片")
            with gr.Column():
                prompt_output = gr.Textbox(label="提示词")
                score_output = gr.Textbox(label="置信度")

        submit_btn = gr.Button("开始分析")

        submit_btn.click(
            fn=image_to_prompt,
            inputs=[image_input],
            outputs=[prompt_output, score_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_webui()
