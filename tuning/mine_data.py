"""
    Extracts the texts from the Chamber of Deputees website to create the bipartite graph for tuning
"""

import numpy as np
import pandas as pd
import concurrent.futures
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao="

propostion_tree = pd.read_csv("../data/arvore-proposicoes.csv", encoding="utf-8")
edges = propostion_tree[["codProposicao", "codProposicaoReferenciada"]].to_numpy()
codes = np.unique(edges[:,0] + edges[:,1])

out = []
CONNECTIONS = 100
TIMEOUT = 10


def load_url(code):
    url = BASE_URL + str(code)
    response = requests.request(method="GET", url=url)
    if (response.status_code != 200):
        return
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "identificacaoProposicao"})
    text = soup.find("span", {"class": "textoJustificado"}).text
    
    return (code, text)


with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
    future_to_url = (executor.submit(load_url, code) for code in codes)
    for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(codes)):
        try:
            data = future.result()
        except Exception as exc:
            data = None
        finally:
            if (data != None):
                out.append(data)

with open("ementas_cod.npy", "wb") as f:
    np.save(f, out)

print(f"Extracted {len(out)} texts")
