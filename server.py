import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
MODEL = "meta-llama/llama-3.1-8b-instruct:free"
SYSTEM = "You are Allba AI. Speak only clear English, no pidgin. Keep replies under 40 words."

@app.get("/")
def home():
    return {"status": "Allba Calls Live Ready"}

@app.get("/chat")
async def chat(text: str):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":text}], "max_tokens": 120}
        )
        return {"reply": r.json()["choices"][0]["message"]["content"]}
