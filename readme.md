# Python FastAPI Project with LangChain for RAG

## Usage
0. Clone the repository to your local machine.

1. Get the your Gemini api-key at [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key) and pass it into config/cfg.yaml ("api_key": "...").

2. Create pythonn env: 
```
python -m venv venv
```
3. Install Requirements:
```
venv\Scripts\activate
python -m pip install -r requirements.txt
```
4. Run the FastAPI server:
```
python main.py
```
5. View the Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

