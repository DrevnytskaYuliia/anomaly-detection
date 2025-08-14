# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
import json

# Initialize FastAPI
app = FastAPI()

# Allow CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "../../static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve HTML UI
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())

# Input schema
class Prompt(BaseModel):
    text: str

# Ollama generation endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"  # you can change to your preferred model

@app.post("/generate")
async def generate_text(prompt: Prompt):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt.text,
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                stream=True
            )
            response.raise_for_status()

            # Combine all response chunks
            full_text = ""
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        full_text += data["response"]

        full_text = full_text.strip()
        if not full_text:
            return JSONResponse(content={"error": "No response from model"}, status_code=500)

        # ✅ Return the generated text
        return {"generated_text": full_text}

    except Exception as e:
        # Log for debugging
        print(f"Error in /generate: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
