import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA=pd.read_csv("dataÀjourCsv.csv")

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

DATA_TEST=DATA.sample(400,random_state=0)
print("La tete :")
print(DATA_TEST.head())
print("les colonnes : ")
print(DATA_TEST.columns)


DATA_TEST["product_type"] = DATA_TEST["product_name"].apply(detect_type)


DATA_TEST["contenu"] = (
    (DATA_TEST['product_name'].astype(str) + " ") * 5 +
    DATA_TEST['product_type'].astype(str) + " " +
    DATA_TEST['about_product'].astype(str)
)


DATA_TEST["contenu"]=DATA_TEST["contenu"].fillna('').str.lower()

def recommander(index_produit , top_n=10) :


    categorie = DATA_TEST.iloc[index_produit]['category_clean']
    idProduit = DATA_TEST.iloc[index_produit]['product_id']
    print("Categorie utilisée :", categorie)
    DATA_FILTRE = DATA_TEST[(DATA_TEST['category_clean'] == categorie)].copy()

    print("Nombre de produits dans DATA_FILTRE :", len(DATA_FILTRE))
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfIDF_matrice= vectorizer.fit_transform(DATA_FILTRE["contenu"])
    similarite =  cosine_similarity(tfIDF_matrice)
     
    indiceLocal = DATA_FILTRE.reset_index(drop=True)
    indiceLocal = indiceLocal[indiceLocal['product_id'] == idProduit].index[0]   
    scores =   list(enumerate(similarite[indiceLocal]))
    scores =  sorted(scores , key=lambda x : x[1]  , reverse=True )
    scores = scores[1 : top_n+1]
    indices = [i[0] for i in scores ]
    
    return DATA_FILTRE.iloc[indices][['product_name','category']]

print("Produit :", DATA_TEST.iloc[119]["product_name"])
print("Category clean :", DATA_TEST.iloc[119]["category_clean"])
print("Category complète :", DATA_TEST.iloc[119]["category"])

print("Produit de base : ",DATA_TEST.iloc[ 10]['product_name'])
print(recommander(10))

#DATA_TEST.to_csv("dataTest.csv",index=False)



