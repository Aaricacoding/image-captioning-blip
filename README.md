# 🖼️ Image Caption Generator — Multi-modal AI Project

> **Author:** Aarica Raj  
> **GitHub:** [@Aaricacoding](https://github.com/Aaricacoding)  
> **Tech Stack:** Python · PyTorch · HuggingFace Transformers · BLIP · Gradio  

---

## 📌 What is this project?

This project builds an **AI-powered Image Captioning System** that automatically generates
human-readable descriptions of any image you upload.

It uses **BLIP (Bootstrapped Language-Image Pretraining)** — a cutting-edge **multi-modal model**
that combines:
- A **Vision Transformer (ViT)** — the "eyes" of the model (CNN-based image encoder)
- A **BERT-style Transformer** — the "language" of the model (text decoder)

Together they understand an image visually and describe it in natural language — just like a human would.

---

## 🧠 How Does It Work? (Architecture)

```
Image Input
    │
    ▼
┌─────────────────────────────┐
│  Vision Encoder (ViT / CNN) │  ← Extracts visual features from image patches
│  (Salesforce BLIP)          │
└─────────────┬───────────────┘
              │  Image Embeddings
              ▼
┌─────────────────────────────┐
│  Text Decoder (Transformer) │  ← Generates caption word by word
│  (BERT-style architecture)  │     using cross-attention on image features
└─────────────┬───────────────┘
              │
              ▼
        "a dog playing in a park"   ← Generated Caption
```

### Key Concepts:
| Concept | Explanation |
|---|---|
| **Multi-modal** | Model works with both image AND text together |
| **Vision Transformer (ViT)** | Splits image into patches, treats them like words |
| **Cross-Attention** | Text decoder "looks at" image features while generating words |
| **Beam Search** | Searches multiple caption options to find the best one |
| **Transfer Learning** | We use a pre-trained BLIP model — no training from scratch needed |

---

## 📁 Project Structure

```
image_captioning_project/
│
├── app.py               ← Main Gradio application (run this!)
├── requirements.txt     ← All Python dependencies
├── README.md            ← This documentation file
│
├── src/
│   └── caption_engine.py  ← Core captioning logic (reusable module)
│
└── examples/            ← Sample images to test the app
    ├── dog.jpg
    └── city.jpg
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Aaricacoding/image-captioning-blip.git
cd image-captioning-blip
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the App
```bash
python app.py
```

### Step 5 — Open in Browser
```
http://localhost:7860
```

> 💡 The first run will automatically download the BLIP model (~1 GB). This only happens once.

---

## 🚀 How to Use

1. Open the app in your browser at `http://localhost:7860`
2. **Upload any image** (JPG, PNG, WEBP supported)
3. Choose **Caption Mode:**
   - `Unconditional` → Model freely generates a caption
   - `Conditional` → Model generates caption guided by a "a photography of" prompt
4. (Optional) Expand **Advanced Settings** to tune:
   - `Max Caption Length` — how long the caption can be
   - `Beam Search Width` — higher = better quality but slower
5. Click **"✨ Generate Caption"**
6. Your caption appears in the output box!

---

## 🧪 Example Outputs

| Image | Generated Caption |
|---|---|
| 🐕 Dog in park | "a dog running in a grassy field on a sunny day" |
| 🌆 City at night | "a city skyline lit up at night with reflections on water" |
| 🍕 Pizza | "a pizza with cheese and toppings on a wooden board" |
| 👩‍💻 Person coding | "a person sitting at a desk working on a laptop computer" |

---

## 🔧 Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Deep learning framework — runs the model |
| **HuggingFace Transformers** | Pre-trained BLIP model + processor |
| **BLIP Model** | Multi-modal vision-language model for captioning |
| **Gradio** | Web UI framework for ML demos |
| **Pillow (PIL)** | Image loading and preprocessing |

---

## 🌐 Model Details

- **Model:** `Salesforce/blip-image-captioning-base`
- **Source:** [HuggingFace Model Hub](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **Parameters:** ~247 million
- **Pre-trained on:** COCO Captions, Visual Genome, Conceptual Captions (130M image-text pairs)
- **License:** BSD-3-Clause

---

## 💡 What I Learned Building This

- How **multi-modal AI** combines vision and language models
- How **Vision Transformers** process images as patches (not pixels)
- How **cross-attention** lets the text decoder "look at" image features
- How to use **HuggingFace Transformers** to load and run pre-trained models
- How to build and deploy ML models using **Gradio**
- How **beam search** improves text generation quality

---

## 🔮 Future Improvements

- [ ] Add support for multiple languages (multilingual captioning)
- [ ] Fine-tune BLIP on a custom dataset (e.g., medical images)
- [ ] Add image upload via URL
- [ ] Deploy on Hugging Face Spaces (free cloud hosting)
- [ ] Add confidence score for generated captions
- [ ] Upgrade to BLIP-2 for even better accuracy

---

## 📤 Deploy to HuggingFace Spaces (Free!)

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space at huggingface.co/spaces
# Upload your files there — it runs automatically!
```

---

## 📄 License

This project is open source under the **MIT License**.  
Feel free to use, modify, and share!

---

## 🙏 Acknowledgements

- [Salesforce Research](https://github.com/salesforce/BLIP) — for the BLIP model
- [HuggingFace](https://huggingface.co) — for the Transformers library
- [Gradio Team](https://gradio.app) — for the amazing UI framework
