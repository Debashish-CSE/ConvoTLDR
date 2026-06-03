# ConvoTLDR 

A web-based text summarization app powered by a fine-tuned **T5 transformer** model. Paste any dialogue or article and get a concise summary instantly.

---

## Demo

![ConvoTLDR UI](https://img.shields.io/badge/Status-Live-brightgreen)

> Enter any conversation or article → click **Summarize** → get a clean, concise summary.

---

## Model

- **Base model:** `t5-small` from HuggingFace Transformers
- **Fine-tuned on:** [SAMSum Dataset](https://huggingface.co/datasets/samsum) — 14,732 training samples of conversational dialogues with human-written summaries
- **Training:** 6 epochs, ~4000 samples, using HuggingFace `Trainer` API
- **Final training loss:** ~0.92

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | HuggingFace Transformers (T5) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Templating | Jinja2 |
| Data | Pandas |
| Deep Learning | PyTorch |

---

## 📁 Project Structure

```
ConvoTLDR/
├── app.py                  # FastAPI backend
├── index.html              # Frontend UI
├── saved_summary_model/    # Fine-tuned T5 model & tokenizer (not pushed)
├── Text_Summarizer.ipynb   # Training notebook
├── requirements.txt        # Dependencies
└── .gitignore
```

---

## Setup & Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Debashish-CSE/ConvoTLDR.git
cd ConvoTLDR
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the model**

The fine-tuned model is not included in this repo due to file size. Either:
- Train it yourself using `Text_Summarizer.ipynb`
- Or load directly from HuggingFace Hub (see notebook)

**4. Run the app**
```bash
python -m uvicorn app:app --reload
```

**5. Open in browser**
```
http://127.0.0.1:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/summarize/` | Returns a summary of the input dialogue |

**POST request body:**
```json
{
  "dialogue": "Your text or conversation here..."
}
```

**Response:**
```json
{
  "summary": "Concise summary of the input."
}
```

---

## Requirements

```
fastapi
uvicorn
transformers
torch
sentencepiece
pydantic
jinja2
pandas
python-multipart
```

---

## Notes

- The model runs on **GPU** if available, falls back to **CPU**
- Input is capped at **512 tokens**; output summary at **150 tokens**
- Uses **beam search** (`num_beams=4`) for better summary quality

---

## Author

**Debashish** — [GitHub](https://github.com/Debashish-CSE)