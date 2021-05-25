from sentence_transformers import SentenceTransformer, SentencesDataset, util, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

model = SentenceTransformer("checkpoint/1000")

tree = pd.read_csv("../data/arvore-proposicoes.csv", encoding="utf-8")
df_corpus = pd.read_csv("../data/proposicao-tema-completo-sem-duplicado.csv", encoding="utf-8")

edges = tree[["codProposicao", "codProposicaoReferenciada"]].to_numpy()
data = df_corpus[["codProposicao", "txtEmenta"]].to_numpy()


codes = data[:,0]
texts = data[:,1]

positives = list()
for edge in tqdm(edges):
    idx1 = np.where(codes == str(edge[0]))[0]
    idx2 = np.where(codes == str(edge[1]))[0]
    if (idx1.shape[0] != 0 and idx2.shape[0] != 0):
        idx1 = idx1[0]
        idx2 = idx2[0]
        
        txt1 = str(texts[idx1])
        txt2 = str(texts[idx2])
        
        positives.append( (txt1[:min(len(txt1),512)], txt2[:min(len(txt2), 512)]) )
print(f"positives {len(positives)}")

negatives = list()
for i in range(len(positives)):
    idx1 = np.random.randint(0, high=df_corpus.shape[0])
    idx2 = np.random.randint(0, high=df_corpus.shape[0])
    
    txt1 = str(df_corpus.iloc[idx1]["txtEmenta"])
    txt2 = str(df_corpus.iloc[idx2]["txtEmenta"])

    txt1 = txt1[:min(len(txt1),512)] 
    txt2 = txt2[:min(len(txt2),512)]
    if ((txt1, txt2) not in negatives and (txt2, txt1) not in negatives and 
        (txt1, txt2) not in positives and (txt2, txt1) not in positives):
        negatives.append((txt1, txt2))
    else:
        i = i-1
print(f"negatives {len(negatives)}")

negatives = np.array(negatives)
positives = np.array(positives)

train_dataset = list()
for x in positives:
    train_dataset.append(InputExample(texts=x, label=1.0))
for x in negatives:
    train_dataset.append(InputExample(texts=x, label=0.0))
    

# Building dataloader and trianing model
#model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, checkpoint_path="./checkpoint", output_path="./model", save_best_model=True)
model.save("./model")
