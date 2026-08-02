# Image-Text Matching using CLIP (PyTorch)

## Overview

This project demonstrates **zero-shot image classification** using **OpenAI CLIP (Contrastive Language–Image Pretraining)**.

Instead of training a classifier, CLIP compares image embeddings with text embeddings and predicts the text description that is most similar to each image.

---

## Features

- Uses the pre-trained **CLIP ViT-B/32** model.
- Loads multiple images.
- Preprocesses images automatically.
- Tokenizes text descriptions.
- Generates image and text embeddings.
- Normalizes feature vectors.
- Computes cosine similarity.
- Finds the best matching text prompt for each image.

---

## Technologies Used

- Python
- PyTorch
- OpenAI CLIP
- Pillow (PIL)

---

## Project Workflow

```
Images
(cat, dog, bird)
        │
        ▼
Image Preprocessing
        │
        ▼
Image Encoder (CLIP)
        │
        ▼
Image Feature Vectors
        │
        ├────────────────────────┐
        │                        │
        ▼                        ▼
Normalize               Text Descriptions
                                │
                                ▼
                           Tokenization
                                │
                                ▼
                         Text Encoder (CLIP)
                                │
                                ▼
                       Text Feature Vectors
                                │
                                ▼
                           Normalize
                                │
                                ▼
                     Cosine Similarity Matrix
                                │
                                ▼
                     Best Matching Description
```

---

## Dataset

The project uses three sample images:

```
data/
│── cat.jpg
│── dog.jpg
│── bird.jpg
```

Text descriptions:

```python
[
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
]
```

---

## Image Preprocessing

Each image undergoes the following preprocessing:

- Resize
- Center Crop
- Convert to Tensor
- Normalize using CLIP statistics
- Add Batch Dimension

The images are concatenated into one batch before being passed to the model.

---

## Model

The project uses the pretrained **ViT-B/32** CLIP model.

```python
model, preprocess = clip.load("ViT-B/32", device=device)
```

The model consists of:

- Image Encoder (Vision Transformer)
- Text Encoder (Transformer)

Both encoders generate feature vectors in the same embedding space.

---

## Feature Extraction

### Image Features

```python
image_features = model.encode_image(image_input)
```

### Text Features

```python
text_features = model.encode_text(text_tokens)
```

Each image and text description is converted into a feature vector.

---

## Feature Normalization

Feature vectors are normalized to unit length.

```python
image_features /= image_features.norm(dim=1, keepdim=True)
text_features /= text_features.norm(dim=1, keepdim=True)
```

Normalization allows the dot product between vectors to represent cosine similarity.

---

## Similarity Computation

Cosine similarity is computed using matrix multiplication.

```python
similarity = 100.0 * image_features @ text_features.T
```

The resulting similarity matrix compares every image with every text description.

Example:

| Image | Cat | Dog | Bird |
|------|------:|------:|------:|
| Cat Image | 98.7 | 14.3 | 8.1 |
| Dog Image | 11.6 | 99.1 | 16.5 |
| Bird Image | 7.8 | 18.4 | 97.9 |

The highest similarity score determines the predicted label.

---

## Prediction

For each image:

1. Find the highest similarity score.
2. Retrieve the corresponding text description.
3. Display the prediction and similarity score.

Example output:

```
cat.jpg -> a photo of a cat and the similarity score is 98.7345

dog.jpg -> a photo of a dog and the similarity score is 99.1042

bird.jpg -> a photo of a bird and the similarity score is 97.8568
```

---

## Project Structure

```
project/
│
├── data/
│   ├── cat.jpg
│   ├── dog.jpg
│   └── bird.jpg
│
├── clip_demo.py
├── README.md
└── requirements.txt
```

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision pillow
pip install git+https://github.com/openai/CLIP.git
```

---

## How to Run

Run the script:

```bash
python clip_demo.py
```

The program will:

- Load the pretrained CLIP model
- Preprocess the images
- Encode images and text
- Compute similarity scores
- Display the best matching text prompt for each image

---

## Future Improvements

- Support custom image folders.
- Accept user-defined text prompts.
- Display top-k predictions.
- Build a graphical user interface using Streamlit or Gradio.
- Extend to image retrieval and semantic search tasks.

---

## Author

**Shahid Farhan KP**
