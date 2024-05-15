from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


app = FastAPI()

model_name = "facebook/nllb-200-3.3B"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class TranslationRequest(BaseModel):
    text: str


def translate_prompt(prompt: str) -> str:
    
    translator = pipeline(
        'translation', model=model, tokenizer=tokenizer,
        src_lang='fon_Latn', tgt_lang='fra_Latn'
    )
    output = translator(prompt, max_length=100)
    translated_text = output[0]['translation_text']
    return translated_text


@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translated_text = translate_prompt(request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to the translation API"}

