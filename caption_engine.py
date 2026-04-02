# ============================================================
#  src/caption_engine.py
#  Reusable captioning module — import this in other scripts
# ============================================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


class CaptionEngine:
    """
    A reusable image captioning engine using the BLIP model.

    Usage:
        engine  = CaptionEngine()
        image   = Image.open("photo.jpg")
        caption = engine.caption(image)
        print(caption)
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        print(f"Loading model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model     = BlipForConditionalGeneration.from_pretrained(model_name)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = self.model.to(self.device)
        self.model.eval()
        print(f"Ready on {self.device.upper()}")

    def caption(
        self,
        image      : Image.Image,
        prompt     : str  = None,
        max_length : int  = 50,
        num_beams  : int  = 5
    ) -> str:
        """
        Generate a caption for a PIL image.

        Args:
            image      : PIL Image object
            prompt     : Optional text prompt to guide the caption
            max_length : Maximum number of tokens in the output
            num_beams  : Beam search width

        Returns:
            str: Generated caption
        """
        image = image.convert("RGB")

        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length    = max_length,
                num_beams     = num_beams,
                early_stopping= True
            )

        return self.processor.decode(output[0], skip_special_tokens=True).capitalize()

    def caption_from_path(self, image_path: str, **kwargs) -> str:
        """Convenience method — pass an image file path directly."""
        image = Image.open(image_path)
        return self.caption(image, **kwargs)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    engine = CaptionEngine()

    # Test with a sample image URL
    import requests
    from io import BytesIO

    url   = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    resp  = requests.get(url)
    image = Image.open(BytesIO(resp.content))

    print("Caption:", engine.caption(image))
    print("Guided: ", engine.caption(image, prompt="a photography of"))
