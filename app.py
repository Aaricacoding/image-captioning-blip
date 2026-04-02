# ============================================================
#  Image Captioning App — BLIP + Gradio
#  Author  : Aarica Raj
#  GitHub  : github.com/Aaricacoding
#  Project : Multi-modal AI (CNN + Transformer)
# ============================================================

import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ── 1. Load Model & Processor ──────────────────────────────
print("Loading BLIP model... (first run may take a minute)")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
model.eval()

print(f"Model loaded on: {device.upper()}")


# ── 2. Caption Generation Function ─────────────────────────
def generate_caption(image, mode, max_length, num_beams):
    """
    Generate a caption for the uploaded image.

    Args:
        image      : PIL Image uploaded by user
        mode       : 'Conditional' (with prompt) or 'Unconditional'
        max_length : Maximum token length for generated caption
        num_beams  : Beam search width (higher = better quality, slower)

    Returns:
        str: Generated caption text
    """
    if image is None:
        return "⚠️ Please upload an image first."

    # Convert to RGB (handles PNG with alpha channel too)
    image = image.convert("RGB")

    if mode == "Conditional (with prompt)":
        text   = "a photography of"
        inputs = processor(image, text, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length = int(max_length),
            num_beams  = int(num_beams),
            early_stopping = True
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.capitalize()


# ── 3. Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    title="Image Caption Generator",
    theme=gr.themes.Soft(),
    css="""
        .header  { text-align: center; padding: 20px 0 10px; }
        .footer  { text-align: center; font-size: 12px; color: #888; margin-top: 10px; }
        .caption { font-size: 18px; font-weight: bold; color: #2c3e50; }
    """
) as demo:

    # Header
    gr.HTML("""
        <div class='header'>
            <h1>🖼️ AI Image Caption Generator</h1>
            <p>Upload any image and let the BLIP multi-modal model describe it!</p>
            <p><i>Model: Salesforce/blip-image-captioning-base &nbsp;|&nbsp;
               Architecture: CNN (ViT) + Transformer (BERT)</i></p>
        </div>
    """)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            image_input = gr.Image(
                type    = "pil",
                label   = "Upload Image",
                height  = 300
            )
            mode = gr.Radio(
                choices = ["Unconditional", "Conditional (with prompt)"],
                value   = "Unconditional",
                label   = "Caption Mode",
                info    = "Unconditional = free caption | Conditional = guided by a prompt"
            )
            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum = 20, maximum = 100, value = 50, step = 5,
                    label   = "Max Caption Length (tokens)"
                )
                num_beams = gr.Slider(
                    minimum = 1, maximum = 10, value = 5, step = 1,
                    label   = "Beam Search Width (higher = better quality)"
                )
            btn = gr.Button("✨ Generate Caption", variant="primary", size="lg")

        # Right column — output
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label       = "Generated Caption",
                placeholder = "Your caption will appear here...",
                lines       = 3,
                elem_classes= ["caption"]
            )
            gr.HTML("<br>")
            gr.Examples(
                examples   = [
                    ["examples/dog.jpg",   "Unconditional",              50, 5],
                    ["examples/city.jpg",  "Conditional (with prompt)",  60, 5],
                ],
                inputs     = [image_input, mode, max_length, num_beams],
                outputs    = [caption_output],
                fn         = generate_caption,
                cache_examples = False,
                label      = "Try these examples"
            )

    # Footer
    gr.HTML("""
        <div class='footer'>
            Built by <b>Aarica Raj</b> &nbsp;|&nbsp;
            <a href='https://github.com/Aaricacoding' target='_blank'>GitHub</a> &nbsp;|&nbsp;
            Powered by HuggingFace Transformers + Gradio
        </div>
    """)

    # Wire button
    btn.click(
        fn      = generate_caption,
        inputs  = [image_input, mode, max_length, num_beams],
        outputs = [caption_output]
    )

# ── 4. Launch ───────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share       = False,   # Set True to get a public URL
        server_name = "0.0.0.0",
        server_port = 7860,
        show_error  = True
    )
