from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer 

import numpy as np
import pandas as pn

DATA=pn.read_excel("dataÀjour.ods",engine="odf")

print(DATA.head())
print(DATA.columns)


DATA["text"]=(DATA["product_name"].fillna("")+" "+DATA["product_name"].fillna("")+" "+DATA["category"].fillna("")+" "+DATA["category"].fillna("")+" "+DATA["about_product"].fillna(""))

MODELE=SentenceTransformer("all-MiniLM-L6-v2",local_files_only=True)
EMB=MODELE.encode(DATA["text"].tolist(),normalize_embeddings=True)


KNN = NearestNeighbors(
    n_neighbors=20 ,
    metric="cosine"

)
KNN.fit(EMB)

def Recommander(i, n=10):
    distances, indices = KNN.kneighbors([EMB[i]], n_neighbors=20)

    base_cat = DATA.iloc[i]["category"]
    base_name = DATA.iloc[i]["product_name"].lower()

    COMPLEMENTARY = {
        "iphone": ["earphone", "airpods", "charger", "case"],
        "smartphone": ["earphone", "charger", "case"],
        "laptop": ["mouse", "keyboard", "bag"],
        "tv": ["soundbar", "wall mount", "remote"]
    }

    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == i:
            continue    
        
        score = 1 - dist

        cat = str(DATA.iloc[idx]["category"]).lower()
        name = str(DATA.iloc[idx]["product_name"]).lower()

        if cat == base_cat.lower():
            score *= 1.2

        for k, vals in COMPLEMENTARY.items():
            if k in base_name:
                if any(v in name for v in vals):
                    score *= 1.5
                break

        results.append((idx, score))

    results.sort(key=lambda x: x[1], reverse=True)

    recommandations=[]

    for idx, score in results[:n]:

        recommandations.append({
            "product_id": str(DATA.iloc[idx]["product_id"]),
            "product_name": str(DATA.iloc[idx]["product_name"]),
            "score": round(float(score),3)
        })


    return recommandations

#   for r, (idx, score) in enumerate(results[:n]):
#        print(f"{r+1}. {DATA.iloc[idx]['product_name']} (score={score:.3f})")

#    return results[:n]

#Recommander(100)

#print("produits de base :" ,DATA.iloc[100]['product_name']) 
#print(Recommander(100))

if __name__ == "__main__":
    print("produits de base :", DATA.iloc[100]['product_name'])
    print(DATA.iloc[81]["product_name"])
    print(Recommander(100))