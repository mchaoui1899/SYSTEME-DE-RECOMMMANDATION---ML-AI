import pandas as pd
import numpy as num

from sentence_transformers import SentenceTransformer 
from    sklearn.metrics.pairwise import cosine_similarity

DATA = pd.read_csv("dataÀjourCsv.csv")
DATA["TEXT"]=(DATA["product_name"].fillna("")+" "+DATA["category"].fillna("")+" "+DATA["about_product"].fillna(""))


DATA["TEXT"]=DATA["TEXT"].str.lower()

MODELE = SentenceTransformer("all-MiniLM-L6-v2")

EM = MODELE.encode(DATA["TEXT"].tolist() , show_progress_bar=True)

matrice_similaritee = cosine_similarity(EM)


def recommander(product_index, top_k=5):
    scores = list(enumerate(matrice_similaritee[product_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    top_items = scores[1:top_k+1]  # on ignore lui-même
    
    return DATA.iloc[[i[0] for i in top_items]][["product_name", "category"]]
print("Produit de base : ",DATA.iloc[77][["product_name", "category", "about_product"]])
print(recommander(77,10))
