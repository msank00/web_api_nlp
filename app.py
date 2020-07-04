from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


set(stopwords.words('english'))

app = Flask(__name__)

@app.route("/")
def my_form():
    return render_template("form.html")

@app.route("/", methods=['POST'])
def sentence_similarity():

    text1 = request.form['text1']
    text2 = request.form['text2']
   
    similarity_score = get_similarity_score(text1, text2)
    
    print(f"similarity score: {similarity_score}")
    
    return render_template('form.html', 
                           sim_score=similarity_score, 
                           text1 = text1, 
                           text2 = text2)
    
def get_similarity_score(text1:str, text2:str):
    text1 = text1.lower()
    text2 = text2.lower()
    stop_words = stopwords.words('english')
    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])
    processed_doc2 = ' '.join([word for word in text2.split() if word not in stop_words])
    
    corpus = [processed_doc1, processed_doc2]
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    
    similarity_score = cosine_similarity(tfidf)[0,1] 
    
    return similarity_score

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=5002, threaded=True)
    