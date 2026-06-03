from fastapi import FastAPI, Request
from pydantic import BaseModel  # pydantic is used to validate the request (our request must be text)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Load the saved model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available(): # for mac
    device = torch.device("mps")
elif torch.cuda.is_available(): # for gpu
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".") # this repository(.) contains the html,css,js code

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str  

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # remove next lines \r\n
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <h1> </p>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue) # preprocess
    
    # tokenize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors = "pt" # tokenizer returns pytorch tensors
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # generate the summary => token ids will be generated
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,   # maximum 150 tokens for our summary
        num_beams = 4,      # returns best output from 4 generated outputs
        early_stopping = True   # model stops as soon as 4 outputs (num_beams=4) are generated
    )
    
    # convert token ids to text => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens = True) # special tokens = EOS, SEP
    return summary

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request}) # UI template as request

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary} # response is send as a json object


# to run the app:python -m uvicorn app:app --reload