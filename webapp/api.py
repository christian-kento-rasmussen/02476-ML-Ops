from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from omegaconf import OmegaConf
from io import BytesIO

from FishEye.predict_model import preprocess_images
from FishEye.predict_model import predict as predict_model_
from FishEye.models.model import FishNN
from PIL import Image
import json

with open("data/processed/label_map.json", "r") as fp:
        label_mapping = json.load(fp)

model = FishNN.load_from_checkpoint("models/epoch=99-step=600.ckpt", cfg=OmegaConf.load("config/config.yaml"))
model.to("cpu")

app = FastAPI()

# Mount the static folder to serve static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Define a route to serve the main HTML file
@app.get("/", response_class=FileResponse)
async def serve_main(request: Request):
    return FileResponse("webapp/static/index.html")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    import torchvision.transforms as transforms

    # Read the image content
    content = await image.read()

    # Convert the image content to a PIL image
    pil_image = Image.open(BytesIO(content))

    # Convert the PIL image to a torch tensor
    tensor_image = transforms.ToTensor()(pil_image)

    tensor_image = preprocess_images([tensor_image])

    # Run the prediction
    predictions = predict_model_(model, tensor_image, label_mapping)

    # Process the image and generate some text
    text = predictions[0] #process_image(image)

    return {"result": text}


    # INstanciate a FishNN model
    # model = FishNN(cfg=OmegaConf.load("config/config.yaml"))

    # Load the best model weights into the model
    import torchvision.transforms as transforms

    

    # Raw data to predict
    image_filenames = [
        "data/raw/NA_Fish_Dataset/Trout/00005.png",
        #                       'data/raw/NA_Fish_Dataset/Red Sea Bream/100_1263.JPG',
        "data/raw/NA_Fish_Dataset/Red Mullet/00017.png",
    ]

    # Load the images from the given paths as tensors
    images = [transforms.ToTensor()(Image.open(filename)) for filename in image_filenames]

    # Run the preprocessing on the images
    images = preprocess_images(images)

    # Specify the label mapping

    

    # Run the prediction
    predictions = predict(model, images, label_mapping)

    print(predictions)