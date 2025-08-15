# As per vLLM .10.0 request responses
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import httpx
import time
import uuid
import os
from enum import Enum

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "codellama:7b")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=256, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

class ChatMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

app = FastAPI(title="OpenAI-Compatible Ollama Wrapper", debug=True)
http_client = httpx.AsyncClient(timeout=TIMEOUT)

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

def create_error_response(message: str, type: str = "server_error", code: int = 500) -> Dict:
    return {
        "error": {
            "message": message,
            "type": type,
            "code": code,
        }
    }

async def make_ollama_request(payload: Dict[str, Any]):
    try:
        response = await http_client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=create_error_response(f"Ollama request failed: {str(e)}")
        )

# --- Root ---
@app.get("/")
def root():
    try:
        return {"status": "ok", "message": "Ollama wrapper is running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "ollama",
            "permission": []
        }]
    }

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    try:
        request_id = f"cmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())

        if isinstance(req.prompt, str):
            prompts = [req.prompt]
        else:
            prompts = req.prompt

        choices = []
        for idx, prompt in enumerate(prompts):
            payload = {
                "model": req.model,
                "prompt": prompt,
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
                "stop": req.stop,
                "stream": False
            }

            result = await make_ollama_request(payload)
            choices.append({
                "text": result.get("response", ""),
                "index": idx,
                "logprobs": None,
                "finish_reason": "stop"
            })

        return {
            "id": request_id,
            "object": "text_completion",
            "created": created_time,
            "model": req.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(" ".join(prompts).split()),
                "completion_tokens": sum(len(c["text"].split()) for c in choices),
                "total_tokens": len(" ".join(prompts).split()) + 
                              sum(len(c["text"].split()) for c in choices)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(str(e))
        )

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        request_id = f"chatcmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())

        # Format messages into prompt
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        
        payload = {
            "model": req.model,
            "prompt": prompt,
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
            "stop": req.stop,
            "stream": False
        }

        result = await make_ollama_request(payload)
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("response", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(result.get("response", "").split()),
                "total_tokens": len(prompt.split()) + len(result.get("response", "").split())
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(str(e))
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)