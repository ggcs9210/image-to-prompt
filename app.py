import torch
from PIL import Image
import clip
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def image_to_prompt(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    candidates = [
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
    text_tokens = clip.tokenize(candidates).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        scores = (image_features @ text_features.T).softmax(dim=-1)
        idx = scores.argmax().item()

    return candidates[idx], float(scores[0][idx].item())

if __name__ == "__main__":
    p, c = image_to_prompt("test.jpg")
    print(p, c)
