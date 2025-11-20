from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

pipe = pipeline("text-generation", model="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", device=-1)

class UserInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Server is running", "model": "HyperCLOVAX-SEED-Text-Instruct-0.5B"}

@app.post("/generate")
def generate_text(user_input: UserInput):
    print("Processing...")
    output = pipe(user_input.text, max_new_tokens=64, do_sample=True)
    return {"result": output[0]['generated_text']}
