from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
import uuid
from faker import Faker
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

@app.route("/queryexpansion/", methods=['POST'])
def query_expansion():

    query_str = request.form['text1']
    do_query_expansion = True

    return render_template(form_type, 
                            query_expand = do_query_expansion, 
                            query = query_str, 
                            result1 = "result 1", 
                            result2 = "result 2", 
                            result3 = "result 3", 
                            result4 = "result 4", 
                            result5 = "result 5")

@app.route("/querysearch/", methods=['POST'])
def query_search():

    query_str = request.form['text2']
    do_sem_search = True

    result = get_search_result(query_str)

    return render_template(form_type, 
                            sem_search = do_sem_search, 
                            doc_id_1 = result["rank1"]["doc_id"], 
                            doc_text_1 = result["rank1"]["doc_text"], 
                            doc_id_2 = result["rank2"]["doc_id"], 
                            doc_text_2 = result["rank2"]["doc_text"], 
                            doc_id_3 = result["rank3"]["doc_id"], 
                            doc_text_3 = result["rank3"]["doc_text"],
                            doc_id_4 = result["rank4"]["doc_id"], 
                            doc_text_4 = result["rank4"]["doc_text"],
                            doc_id_5 = result["rank5"]["doc_id"], 
                            doc_text_5 = result["rank5"]["doc_text"])

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
    