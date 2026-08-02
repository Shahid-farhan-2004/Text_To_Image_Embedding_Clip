import torch
import clip
from PIL import Image
import os

device="cuda" if torch.cuda.is_available() else "cpu"
model,preprocess=clip.load("ViT-B/32",device=device)

image_paths=["data/cat.jpg","data/dog.jpg","data/bird.jpg"]
images=[preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
image_input=torch.cat(images,dim=0)

text_descriptions=["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_tokens=clip.tokenize(text_descriptions).to(device)

with torch.no_grad():
  image_features=model.encode_image(image_input)
  text_features=model.encode_text(text_tokens)
  #normalize
  image_features/=image_features.norm(dim=1,keepdim=True)
  text_features/=text_features.norm(dim=1,keepdim=True)
  #similarity
  similarity=(100.0*image_features@text_features.T)

for i,path in enumerate(image_paths):
  best_match=similarity[i].argmax().item()
  print(f"{os.path.basename(path)} -> {text_descriptions[best_match]} and the similarity score is {similarity[i][best_match]:.4f}")
