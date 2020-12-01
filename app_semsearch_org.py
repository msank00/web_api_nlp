from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
import uuid
import json
from faker import Faker
import requests as req
fk = Faker()

set(stopwords.words('english'))

app = Flask(__name__)

form_type = "form_boost_semsearch.html"

@app.route("/")
def my_form():
    return render_template(form_type)

@app.route("/reset/", methods=["POST"])
def reset_page():
    return render_template(form_type)

def get_query_suggestion(query: str):

    print(query)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    url_query_emb_api = "http://scl000106748.sccloud.swissre.com:8000/get_query_expansion"
    payload = {"string": query}

    response = req.post(
        url_query_emb_api, data=json.dumps(payload), headers=headers
    )

    print("here i am ...")
    result = json.loads(response.content)

    return result

def get_documents_by_semantic_search(query: str):

    print(query)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    url_query_emb_api = "http://scl000106748.sccloud.swissre.com:8081/semsearch"
    payload = {"query": query}

    response = req.post(
        url_query_emb_api, data=json.dumps(payload), headers=headers
    )

    print("here i am ...")
    result = json.loads(response.content)

    return result


@app.route("/queryexpansion/", methods=['POST'])
def query_expansion():

    query_str = request.form['text1']
    print(query_str)
    result = get_query_suggestion(query_str)
    ngram_exp_query = result["suggestions"]["ngram"]
    words_exp_query = result["suggestions"]["word"]
    sent_exp_query = result["suggestions"]["sentence"]
    print(words_exp_query)
    do_query_expansion = True

    return render_template(form_type, 
                            query_expand = do_query_expansion, 
                            query = query_str, 
                            result1 = ngram_exp_query[0], 
                            result2 = ngram_exp_query[1], 
                            result3 = ngram_exp_query[2], 
                            result4 = ngram_exp_query[3], 
                            result5 = ngram_exp_query[4],
                            words_suggestion_1 = words_exp_query[0],
                            words_suggestion_2 = words_exp_query[1],
                            words_suggestion_3 = words_exp_query[2],
                            sent_suggestion_1 = sent_exp_query[0], 
                            sent_suggestion_2 = sent_exp_query[1],
                            sent_suggestion_3 = sent_exp_query[2],
                            sent_suggestion_4 = sent_exp_query[3],
                            sent_suggestion_5 = sent_exp_query[4])

@app.route("/querysearch/", methods=['POST'])
def query_search():

    query_str = request.form['text2']
    do_sem_search = True

    # result = get_search_result(query_str)
    result = get_documents_by_semantic_search(query_str)
    print(result["documents"][0]["doc_id"])

    query_emb = list(np.random.rand(20))
    payload_template_dummy = {
        "query": {
            "elastiknn_nearest_neighbors": {
                "field": "content.contentVector",                     
                "vec": {                               
                    "values": query_emb+["..."]
                },
                "model": "lsh", 
                "similarity": "angular",
                "candidates": 50                   
            }
        }
    }

    return render_template(form_type, 
                            sem_search = do_sem_search, 
                            query = query_str,
                            payload_template_html = payload_template_dummy,
                            doc_id_1 = f"Doc id: {result['documents'][0]['doc_id']}", 
                            doc_text_1 = result["documents"][0]["content"], 
                            doc_id_2 = f"Doc id: {result['documents'][1]['doc_id']}",  
                            doc_text_2 = result["documents"][1]["content"], 
                            doc_id_3 = f"Doc id: {result['documents'][2]['doc_id']}",  
                            doc_text_3 = result["documents"][2]["content"],
                            doc_id_4 = f"Doc id: {result['documents'][3]['doc_id']}",  
                            doc_text_4 = result["documents"][3]["content"],
                            doc_id_5 = f"Doc id: {result['documents'][4]['doc_id']}",  
                            doc_text_5 = result["documents"][4]["content"])


@app.route("/keywordsearch/", methods=['POST'])
def keyword_search():

    query_str = request.form['text3']
    do_key_search = True

    # result = get_search_result(query_str)
    result = get_documents_by_keyword_search(query_str)
    print(len(result["documents"]))
    
    return render_template(form_type, 
                            key_search = do_key_search, 
                            query = query_str,
                            doc_id_1 = f"Doc id: {result['documents'][0]['doc_id']}", 
                            doc_text_1 = result["documents"][0]["content"], 
                            doc_id_2 = f"Doc id: {result['documents'][1]['doc_id']}", 
                            doc_text_2 = result["documents"][1]["content"],
                            doc_id_3 = f"Doc id: {result['documents'][2]['doc_id']}",  
                            doc_text_3 = result["documents"][2]["content"],
                            doc_id_4 = f"Doc id: {result['documents'][3]['doc_id']}",  
                            doc_text_4 = result["documents"][3]["content"],
                            doc_id_5 = f"Doc id: {result['documents'][4]['doc_id']}",  
                            doc_text_5 = result["documents"][4]["content"])

    

def get_documents_by_keyword_search(query):
    
    print(query)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    url_query_emb_api = "http://scl000106748.sccloud.swissre.com:8081/keywordsearch"
    payload = {"query": query}

    response = req.post(
        url_query_emb_api, data=json.dumps(payload), headers=headers
    )

    print("here i am ...")
    result = json.loads(response.content)

    return result

def get_search_result(query: str):

    result = {}
    for i in range(5):
        rank_no = i+1
        result[f"rank{rank_no}"]={} 
        result[f"rank{rank_no}"]["doc_id"] = f"Rank {rank_no}, Doc-ID: {str(uuid.uuid1())}"
        result[f"rank{rank_no}"]["doc_text"] = fk.text()

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=8085, threaded=True)
    
