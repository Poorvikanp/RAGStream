# File QA RAG Chatbot

This repository contains a Streamlit-based RAG (Retrieval-Augmented Generation) app adapted from the notebook `End to end RAG Application Development with LangChain and Streamlit.ipynb`.

Files created:
- `app.py`: main Streamlit application (extracted from notebook)
- `requirements.txt`: Python dependencies
- `ngrok_tunnel.py`: helper to open an ngrok tunnel
- `run_app.bat`: helper to start Streamlit on Windows

Quick start:
1. Create a `api_keys.yml` with your keys, e.g.: 

```yaml
OPENAI_API_KEY: "sk-..."
NGORK_AUTH_TOKEN: "ngrok_..."
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app (Windows):

```powershell
.\run_app.bat
```

4. (Optional) Start ngrok tunnel:

```bash
python ngrok_tunnel.py
```

Notes:
- The code is a direct extraction from the notebook; you may need to adjust package versions or import paths depending on your environment.
- The app expects uploaded PDF files via the Streamlit sidebar. The vector DB will persist in `./chroma_db`.
