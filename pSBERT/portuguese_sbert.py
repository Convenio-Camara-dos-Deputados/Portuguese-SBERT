
from sentence_transformers import SentenceTransformer, models, InputExample
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine

class pSBERT:
    def __init__(self, huggingface_model):
        self.model = SentenceTransformer(huggingface_model)
        return

    def encode(self, sentences):
        
        return
    
    def fit(self):
        return