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

form_type = "form_boost_v3.html"

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
    exp_query = result["suggestions"]["ngram"]
    print(exp_query)
    do_query_expansion = True

    return render_template(form_type, 
                            query_expand = do_query_expansion, 
                            query = query_str, 
                            result1 = exp_query[0], 
                            result2 = exp_query[1], 
                            result3 = exp_query[2], 
                            result4 = exp_query[3], 
                            result5 = exp_query[4])

@app.route("/querysearch/", methods=['POST'])
def query_search():

    query_str = request.form['text2']
    do_sem_search = True

    # result = get_search_result(query_str)
    result = get_documents_by_semantic_search(query_str)
    print(result["documents"][0]["doc_id"])

    return render_template(form_type, 
                            sem_search = do_sem_search, 
                            doc_id_1 = result["documents"][0]["doc_id"], 
                            doc_text_1 = result["documents"][0]["content"], 
                            doc_id_2 = result["documents"][1]["doc_id"], 
                            doc_text_2 = result["documents"][1]["content"], 
                            doc_id_3 = result["documents"][2]["doc_id"], 
                            doc_text_3 = result["documents"][2]["content"],
                            doc_id_4 = result["documents"][3]["doc_id"], 
                            doc_text_4 = result["documents"][3]["content"],
                            doc_id_5 = result["documents"][4]["doc_id"], 
                            doc_text_5 = result["documents"][4]["content"])

def get_search_result(query: str):

    result = {}
    for i in range(5):
        rank_no = i+1
        result[f"rank{rank_no}"]={} 
        result[f"rank{rank_no}"]["doc_id"] = f"Rank {rank_no}, Doc-ID: {str(uuid.uuid1())}"
        result[f"rank{rank_no}"]["doc_text"] = fk.text()

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=8090, threaded=True)
    