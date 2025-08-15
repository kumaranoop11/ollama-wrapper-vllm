# ollama-wrapper-vllm
A FastAPI-based OpenAI-compatible API wrapper is built around **Ollama** and **vLLM** to facilitate seamless integration of LLM models, enabling deployment and serving either locally (via Ollama) or remotely (via vLLM) using a consistent API interface.

Note: Use vLLM v0.10.0

## 🚀 Features
- REST API with FastAPI
- Async endpoints for LLM inference
- Poetry-based dependency management
- Easily configurable for local or containerized deployment

---

## 📂 Project Structure
```
.
├── LICENSE
├── README.md
├── poetry.lock
├── pyproject.toml
└── src
    ├── __init__.py
    └── main.py          # FastAPI entrypoint
```

---

## 🛠 Prerequisites
- Python **3.9+**
- [Poetry](https://python-poetry.org/docs/#installation) installed
- [Ollama](https://ollama.ai/) or a running [vLLM](https://github.com/vllm-project/vllm) instance
- (Optional) Docker for containerized deployment

---

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/kumaranoop11/ollama-wrapper-vllm.git
cd ollama-wrapper-vllm
```

Install dependencies:
```bash
poetry install
```

---

## ▶️ Running the API

### Development
```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production
```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## ⚙️ Configuration
You can set environment variables (e.g., model name, host, port) using a `.env` file in the project root.

Example `.env`:
```env
MODEL_NAME=llama2
OLLAMA_HOST=http://localhost:11434
```

---

## 📡 API Endpoints

| Method | Endpoint       | Description |
|--------|---------------|-------------|
| GET    | `/`           | Health check or welcome message |
| POST   | `/generate`   | Generate text using the configured LLM |
| POST   | `/chat`       | (If implemented) Chat-style conversation API |

*(Check `src/main.py` for the actual implemented routes.)*

---

## 🧪 Testing
```bash
poetry run pytest
```

---

## 🐳 Docker (Optional)
Build image:
```bash
docker build -t ollama-wrapper-vllm .
```
Run container:
```bash
docker run -p 8000:8000 ollama-wrapper-vllm
```

---

## 📜 License
This project is licensed under the terms of the **MIT License**. See [LICENSE](LICENSE) for details.
