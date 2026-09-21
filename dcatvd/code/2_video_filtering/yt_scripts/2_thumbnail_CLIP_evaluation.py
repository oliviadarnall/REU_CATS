from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import pandas as pd

#Configure the Channel, paths, and model

CHANNEL_ID = "UCgkOus3F_UXYcLMBIf39L3Q"

MODEL_NAME = "openai/clip-vit-base-patch32"

THUMBNAIL_DIR = f"thumbnails/{CHANNEL_ID}"

OUTPUT_CSV = f"metadata/{CHANNEL_ID}/thumbnail_results.csv"

MAX_IMAGES = 1000

CAT_THRESHOLD = 0.50

#Load in the model

model = CLIPModel.from_pretrained(MODEL_NAME)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

#Set the labels

labels = [
    "a photo of a cat",
    "a photo without a cat"
]

#Get the image files from the thumbnail directory

image_files = [
    f for f in os.listdir(THUMBNAIL_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_files = image_files[:MAX_IMAGES]

results = []

#Run inference on the thumbnail images

for filename in image_files:
    image_path = os.path.join(THUMBNAIL_DIR, filename)

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        cat_prob = probs[0].item()
        no_cat_prob = probs[1].item()
        match = cat_prob >= CAT_THRESHOLD
        results.append({
            "thumbnail": filename,
            "contains_cat_probability": round(cat_prob, 4),
            "no_cat_probability": round(no_cat_prob, 4),
            "match": match
        })

        print(
            f"{filename} -> "
            f"cat={cat_prob:.3f}, "
            f"no_cat={no_cat_prob:.3f}, "
            f"match={match}"
        )

    except Exception as e:
        print(f"ERROR processing {filename}")
        print(e)

#Save the results in a Pandas dataframe and export to .csv

os.makedirs("metadata", exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved results to:")
print(OUTPUT_CSV)