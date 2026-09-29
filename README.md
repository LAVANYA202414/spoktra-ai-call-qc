# Spoktra AI - Call Quality Control with DistilBERT

Fine-tuned DistilBERT for detecting policy violations in customer support calls. 
Built end-to-end ML pipeline with freezing/unfreezing strategy.

**Results:** 89% F1-score, 18% reduction in false positives
**Stack:** Python, PyTorch, DistilBERT, FastAPI, MySQL

### Run
pip install -r requirements.txt
python main.py

### Pipeline
config/ -> paths & hyperparams
src/data_ingestion.py -> read raw calls
src/data_preprocessing.py -> clean & structure
src/train.py -> DistilBERT fine-tuning with class weights
src/inference.py -> predict_call()
