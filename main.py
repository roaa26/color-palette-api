# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image

app = FastAPI()

# تحويل RGB إلى HEX
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# استخراج الألوان باستخدام KMeans
def extract_palette(image, num_colors=6):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(image_rgb, (150, 150))
    flat_image = small_image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(flat_image)

    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    
    sorted_indices = np.argsort(-counts)
    colors = centers[sorted_indices]
    proportions = (counts[sorted_indices] / sum(counts) * 100).round(2).tolist()
    hex_colors = [rgb_to_hex(color) for color in colors]

    return hex_colors, proportions

# API endpoint
@app.post("/color-palette/")
async def get_palette(file: UploadFile = File(...)):
    content = await file.read()
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File must be an image."})
    image = Image.open(BytesIO(content)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    hex_colors, proportions = extract_palette(image_np)

    return JSONResponse(content={
        "palette": hex_colors,
        "proportions (%)": proportions
    })
