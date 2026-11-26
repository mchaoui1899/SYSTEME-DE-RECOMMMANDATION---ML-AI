import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = pd.read_csv("dataÀjourCsv.csv")

DATA["category_clean"] = DATA["category"].astype(str).apply(lambda x: x.split("|")[-1])

def detect_type(text):
    text = str(text).lower()

    if "watch" in text or "smartwatch" in text:
        return "smartwatch"
    if "mouse" in text:
        return "mouse"
    if "keyboard" in text:
        return "keyboard"
    if "cable" in text:
        return "cable"
    if "charger" in text:
        return "charger"
    if "headphone" in text or "earbuds" in text:
        return "audio"
    return "other"


DATA_TEST = DATA.sample(400, random_state=0).reset_index(drop=True)

DATA_TEST["product_type"] = DATA_TEST["product_name"].apply(detect_type)

DATA_TEST["contenu"] = (
    (DATA_TEST["product_name"].astype(str) + " ") * 6 +
    (DATA_TEST["product_type"].astype(str) + " ") * 2 +
    DATA_TEST["about_product"].astype(str).fillna("")
)

DATA_TEST["contenu"] = DATA_TEST["contenu"].str.lower()


def recommander(index_produit, top_n=10):

    # Produit de Base 
    produit_ref = DATA_TEST.iloc[index_produit]
    categorie = produit_ref["category_clean"]
    idProduit = produit_ref["product_id"]

    print("\nProduit utilisé :", produit_ref["product_name"])
    print("Categorie :", categorie)

    # filtre correct
    DATA_FILTRE = DATA_TEST.copy()

    print("Nombtre des produits filtrés :", len(DATA_FILTRE))

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(DATA_FILTRE["contenu"])
    similarity = cosine_similarity(tfidf_matrix)

    indiceLocal = DATA_FILTRE[DATA_FILTRE["product_id"] == idProduit].index[0]

    scores = list(enumerate(similarity[indiceLocal]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_scores = scores[1:top_n+1]
    indices = [i[0] for i in top_scores]

    return DATA_FILTRE.iloc[indices][["product_name", "category"]]


# TEST
index_test = 399

print("\n LE Produit de base :", DATA_TEST.iloc[index_test]["product_name"])
print(recommander(index_test , 10))