from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from arabert.preprocess import ArabertPreprocessor

app = FastAPI()

arabert_prep = ArabertPreprocessor(model_name="araelectra-base-discriminator")

qa_pipeline = pipeline(
    "question-answering",
    model="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA",
    tokenizer="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"
)

class QARequest(BaseModel):
    context: str
    question: str

@app.post("/qa")
def answer_qa(request: QARequest):
    context_processed = arabert_prep.preprocess(request.context)
    question_processed = arabert_prep.preprocess(request.question)

    result = qa_pipeline({
        "context": context_processed,
        "question": question_processed
    })
    return {"answer": result}
