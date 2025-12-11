import gradio as gr
from webui import image_to_prompt, CANDIDATES, model, preprocess, device  # reuse logic

def build_spaces_demo():
    with gr.Blocks(title="Image-to-Prompt WebUI - HuggingFace Space") as demo:
        gr.Markdown(
            """
            # Image-to-Prompt WebUI (HuggingFace Space)
            Upload an image and get an AI-generated prompt.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="numpy",
                    label="Upload image"
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                prompt_output = gr.Textbox(
                    label="Prompt",
                    lines=2
                )
                score_output = gr.Textbox(
                    label="Score",
                    lines=1
                )

        analyze_btn.click(
            fn=image_to_prompt,
            inputs=[image_input],
            outputs=[prompt_output, score_output],
        )

    return demo

demo = build_spaces_demo()

if __name__ == "__main__":
    demo.launch()
