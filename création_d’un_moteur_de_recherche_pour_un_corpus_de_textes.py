
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')
nltk.download('stopwords')

"""### Corpus de documents"""

data = pd.read_csv("/content/drive/MyDrive/dataset/bbc_dataset.csv")
data.head()

data.drop_duplicates(inplace=True)
print(f"il y a {data.shape[0]} documents")
print(data['Label'].value_counts())

"""### Nettoyage des documents"""

stemmer = PorterStemmer()
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words_stps = [word for word in words if word not in stopwords.words('english')]
    stemmed_words = [stemmer.stem(word) for word in words_stps]
    return " ".join(stemmed_words)

corpus=data.copy()
corpus['Document'] = corpus['Document'].apply(clean_text)
corpus.head()

"""### Indexation TF-IDF"""

vectorizer = TfidfVectorizer()
term_doc_matrix = vectorizer.fit_transform(corpus['Document'])
tfidf_matrix = term_doc_matrix.toarray()
feature_names = vectorizer.get_feature_names_out()
tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=feature_names)
tfidf_matrix.head()

"""### Topic Modeling (LSA)"""

svd = TruncatedSVD(n_components=5, random_state=42)
lsi_matrix = svd.fit_transform(term_doc_matrix)
def get_top_words(svd, feature_names, n_top_words=10):
    topic_words = []
    for topic_idx, topic in enumerate(svd.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_words.append(top_words)
    return topic_words
topic_words = get_top_words(svd, feature_names)
for topic_idx, topic in enumerate(topic_words):
    print(f"Topic {topic_idx + 1}: {', '.join(topic)}")

"""### Requête


1.   Analyse de la requête
2.   Détection des topics
3.   Documents pertinents


"""

def find_similar_docs(query_vector, lsi_matrix):
    similarities = cosine_similarity(query_vector, lsi_matrix).flatten()
    most_similar_indices = similarities.argsort()[:-11:-1]
    return most_similar_indices,similarities

#sanstopicmodeling
def find_similar_docssans(query_vector, tfidf_matrix):
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_indices = similarities.argsort()[:-11:-1]
    return most_similar_indices,similarities

query_text="Microsoft"
clean_text(query_text)
query_vector = vectorizer.transform([query_text])
query_vector = svd.transform(query_vector)
similar_docs,similarities = find_similar_docs(query_vector, lsi_matrix)
print(f"Requête : {query_text}")
for doc_index in similar_docs:
         print(f"  Document {doc_index} : {similarities[doc_index]} : {data['Document'].iloc[doc_index]}")

query_text="Microsoft"
clean_text(query_text)
query_vector = vectorizer.transform([query_text])
similar_docs,similarities = find_similar_docssans(query_vector, tfidf_matrix)
print(f"Requête : {query_text}")
for doc_index in similar_docs:
         print(f"  Document {doc_index} : {similarities[doc_index]} : {data['Document'].iloc[doc_index]}")

"""### Implémentation du moteur de recherche"""

def find_similar_docs_2(query_vector, lsi_matrix):
    similarities = cosine_similarity(query_vector, lsi_matrix).flatten()
    most_similar_indices = similarities.argsort()[::-1]
    return most_similar_indices,similarities

!pip install flask pyngrok

!pip install flask-ngrok

file='/content/drive/MyDrive/templates'
files='/content/drive/MyDrive/static'

def query_process(query):
  query_text=clean_text(query)
  query_vector = vectorizer.transform([query_text])
  query_vector = svd.transform(query_vector)
  return query_vector

from flask import Flask, render_template,request
from pyngrok import ngrok,conf
from flask_ngrok import run_with_ngrok
conf.get_default().auth_token = "2qRwqyoL3XKciS6OYqN8Z76Vjpo_2a1uTCtmKHUNkaMVd7Hbq"

app = Flask(__name__,template_folder=file,static_folder=files)
run_with_ngrok(app)
@app.route('/')
def home():
    return render_template("index_search.html")
@app.route('/search',methods=['POST'])
def search():
  if request.method == 'POST':
    query = request.form['query']
    query_vector=query_process(query)
    similar_docs,similarities = find_similar_docs_2(query_vector, lsi_matrix)
    result_dict={}
    doc_number=0
    for doc_index in similar_docs:
      if similarities[doc_index]>0.88:
        doc_number+=1
        result_dict[doc_index] = {
        'article': data['Article'].iloc[doc_index],
        'document': data['Document'].iloc[doc_index]
         }
    return render_template("index_search.html", results=result_dict,requete=query,results_number=doc_number)
  return render_template("index_search.html")

import time
if __name__ == '__main__':
    ngrok.connect(5000)
    time.sleep(5)
    app.run()