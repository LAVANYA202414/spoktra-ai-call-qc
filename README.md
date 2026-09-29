# Spoktra AI - Call Quality Control with DistilBERT
Fine-tuned DistilBERT for detecting policy violations in customer support calls.
Built end-to-end ML pipeline with freezing/unfreezing strategy.

**Results:** 89% F1-score, 18% reduction in false positives
**Stack:** Python, PyTorch, DistilBERT, FastAPI, MySQL, Transformers

### Metrics
| Metric | Before (Manual) | After (DistilBERT) |
| Accuracy | 71% | 89% |
| False Positives | 32% | 14% |

### How to Run
pip install -r requirements.txt
python main.py

### API (Add this to get Backend jobs)
uvicorn src.inference:app --reload
POST /predict {"conversation": [...]}

### Pipeline
config/ -> paths & hyperparams
src/data_ingestion.py -> read raw calls
src/data_preprocessing.py -> clean & structure
src/train.py -> DistilBERT fine-tuning with class weights + freezing BERT 3 epochs
src/inference.py -> predict_call()
