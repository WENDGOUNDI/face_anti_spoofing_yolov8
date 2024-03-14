# Libraries Importation
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO

# Write app description
description = """
    model: pretrained yolov8, 'yolov8n-cls.pt'
    dataset: Large Crowdedcollected Face Anti-Spoofing
"""

# app intialization
app = FastAPI(
    title="YoloV8 Face AntiSpoofing",
    description=description,
    summary="The goal of this app is to classify face images as real or spoof.",
    version="0.0.1",
    contact={
        "name": "SAVADOGO ABDOUL",
        "github": "https://github.com/wendgoundi",
        "linkedin": "https://tw.linkedin.com/in/wendgoundi-abdoul-rasman%C3%A9-savadogo",
    }
    )

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# Define a function for prediction
def predImage(image):
    # Load a model
    model = YOLO('./runs/classify/train2/weights/best.pt')  # load a custom model
    # Predict with the model
    results = model(image, verbose=False )  # predict on an image
    for result in results:
        probs = result.probs  # Probs object for classification outputs
        if probs.top1 == 1:
            return f"Predicted Label: {model.names[probs.top1]}"
        elif probs.top1 == 0:
            return f"Predicted Label: {model.names[probs.top1]}"


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    print("Image Loaded Successfully!")
    image = read_imagefile(await file.read())
    prediction = predImage(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)