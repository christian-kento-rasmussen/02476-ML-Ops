from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (if needed)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "https://yourfrontenddomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder to serve static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Define a route to serve the main HTML file
@app.get("/", response_class=FileResponse)
async def serve_main(request: Request):
    return FileResponse("webapp/static/index.html")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Process the image and generate some text
    text = "Mullet fish" #process_image(image)

    return {"prediction": text}
