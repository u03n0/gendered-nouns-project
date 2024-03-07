from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# class PredictionRequest(BaseModel):
#     text: str

# class PredictionResponse(BaseModel):
#     result: str

# # Load the pre-trained model
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# num_labels = 2  # Assuming binary classification (e.g., male/female)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# # Load saved model parameters
# folder_path = 'saved_models'
# file_name = 'GenderBert_params.pth'
# file_path = Path(folder_path) / file_name

# if file_path.exists():
#     print('Model was found')
    
#     # Load the state dictionary
#     state_dict = torch.load(file_path)

#     # Create an instance of GenderBert
#     clf = GenderBert(
#         embedding_dim=768,
#         hidden_dim=256,
#         vocab_size=32000,
#         tagset_size=num_labels,
#         epochs=5,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         model=model
#     )

#     # Load the state dictionary into the model, allowing for non-strict loading
#     clf.load_state_dict(state_dict, strict=False)
#     print('AYYYOOO IT WORKED!!!!')

# else:
#     raise HTTPException(status_code=500, detail="Model not found")

# Define API endpoint for prediction
@app.get("/")
def root():
    return {"hello": "World"}
# async def predict(request: PredictionRequest):
#     text = request.text
#     input_ids = tokenizer.encode(text, return_tensors='pt', max_length=64)
#     attention_mask = torch.ones_like(input_ids)

#     # Make prediction
#     with torch.no_grad():
#         prediction = clf.model(input_ids, attention_mask=attention_mask)
#         logits = prediction.logits
#         _, predicted_class = torch.max(logits, 1)

#     return {"result": "male" if predicted_class.item() == 1 else "female"}
