from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from models import GenderBert

CONFIGURATION = {
    'bert': {
        'uses_hf': True,
        'model_name': 'bert-base-uncased',
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertForSequenceClassification.from_pretrained('bert-base-uncased'),
        'max_length': 40,
        'batch_size': 32,
        'custom_model':GenderBert,
        'epochs': 10
    },

    'cnn': {
        'uses_hf': False,
        
    }
}
