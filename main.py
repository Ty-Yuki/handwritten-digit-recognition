import asyncio
import base64
import io
import os

import nest_asyncio  # type: ignore
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pyngrok import ngrok  # type: ignore
from uvicorn import Config, Server

from app.model_utils import load_or_train_model
from app.visualization import visualize_activations

MODEL_PATH = os.getenv("MODEL_PATH", "models/mnist_cnn.h5")
NGROK_TOKEN = os.getenv("NGROK_TOKEN")
PORT = 8000

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def load_model_on_startup() -> None:
    """起動時にモデルをロード"""
    app.state.model = load_or_train_model(MODEL_PATH)


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """手書き数字画像を受け取り、予測と中間層画像を返す

    Args:
        request (Request): リクエストオブジェクト

    Returns:
        JSONResponse: 予測ラベルと中間層画像リスト
    """
    data = await request.json()
    image_bytes = base64.b64decode(data["image"].split(",")[1])
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    arr = np.array(img)
    model = app.state.model
    pred, activation_imgs = visualize_activations(arr, model)
    return JSONResponse({"pred": pred, "activation_imgs": activation_imgs})


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """トップページのHTMLを返す

    Returns:
        HTMLResponse: HTMLコンテンツ
    """
    return templates.TemplateResponse("index.html", {"request": request})


def main() -> None:
    """FastAPIサーバーをngrokで公開して起動する"""
    nest_asyncio.apply()
    public_url = ngrok.connect(PORT)
    print(f"🔗 公開URL: {public_url}")
    config = Config(app=app, host="0.0.0.0", port=PORT, log_level="info")
    server = Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
