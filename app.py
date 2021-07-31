from flask import Flask, render_template,request,redirect, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
app = Flask(__name__)


df = pd.read_csv('datasets/main_data.csv')
file = open('datasets/categories.json')
categories = json.load(file)
file.close()

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['meta_data'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

all_products_with_index = {}
for i in range(df.shape[0]):
    if all_products_with_index.get(df['product_name_val'].iloc[i]) == None:
        all_products_with_index[df['product_name_val'].iloc[i]] = i

def get_recommendation(x):
    idx = all_products_with_index[x]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:31]
    product_indices = [i[0] for i in sim_scores]
    return product_indices

@app.route('/', methods = ['GET' , "POST"])
def home():
    return render_template('index.html',categories=categories)
    

@app.route('/category/<string:c_name>')
def view_all_products(c_name):
    all_products = df[df['product_category'] == c_name][['product_name','product_name_val','img']]
    all_products = all_products.reset_index(drop=True)
    return render_template('allProducts.html',all_products=all_products,c_name=c_name)

@app.route('/product/<string:p_val>')
def product_info(p_val):
    product_indices = get_recommendation(p_val)
    if len(product_indices) > 10:
        product_indices = product_indices[:12]
    rec_products = df.iloc[product_indices]
    rec_products = rec_products.reset_index(drop=True)
    return render_template('product.html',rec_products=rec_products)

if __name__ == '__main__':
    app.run(debug=True) 