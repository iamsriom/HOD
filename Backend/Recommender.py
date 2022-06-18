#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:43:38 2022

@author: sriomchakrabarti
"""

import re
import warnings
from ast import literal_eval

# %matplotlib inline   #ignored for now
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

warnings.simplefilter("ignore")


ordersExport = pd.read_csv("datasets/orders_export.csv")
productsExport = pd.read_csv("datasets/products_export_1.csv")


df_order = ordersExport[["Name", "Lineitem quantity", "Lineitem name", "Lineitem sku"]]

df_product = productsExport[
    ["Handle", "Title", "Custom Product Type", "Tags", "Variant SKU"]
]

final_product = df_product[df_product["Tags"].isnull() == False]

final_product["Handle"] = final_product["Handle"].apply(
    lambda x: [str.lower(i.replace("-", "")) for i in x]
)

final_product["Handle"] = final_product["Handle"].apply(lambda x: "".join(x))

final_product["Title"] = final_product["Title"].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x]
)

final_product["Title"] = final_product["Title"].apply(lambda x: "".join(x))

final_product["Tags"] = final_product["Tags"].apply(
    lambda x: [str.lower(i.replace("_", "")) for i in x]
)
final_product["Tags"] = final_product["Tags"].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x]
)

final_product["Tags"] = final_product["Tags"].apply(lambda x: "".join(x))
final_product["Tags"] = final_product["Tags"].apply(
    lambda x: [str.lower(i.replace(",", " ")) for i in x]
)
final_product["Tags"] = final_product["Tags"].apply(lambda x: "".join(x))

final_product["description"] = (
    final_product["Handle"] + " " + final_product["Title"] + " " + final_product["Tags"]
)
final_product["description"] = final_product["description"].fillna("")


def clean_description(text):
    text = re.sub("'", "", text)
    #     text = re.sub("[^a-zA-Z]"," ",text)
    text = " ".join(text.split())
    text = text.lower()
    return text


final_product["clean_description"] = final_product["description"].apply(
    lambda x: clean_description(x)
)


count = CountVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english"
)
count_matrix = count.fit_transform(final_product["description"])


cosine_sim = cosine_similarity(count_matrix, count_matrix)

final_product = final_product.reset_index()

sku = final_product["Variant SKU"]


indices = pd.Series(final_product.index, index=final_product["Title"])


def get_recommendations(title):
    idx = indices[title]

    # It will get the title with its index
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    # [2502, 7535, 4702, 889, 437]
    product_indices = [i[0] for i in sim_scores]

    return sku.iloc[product_indices]


def get_sku():
    """
    function to get store key units value based on the recommendation model
    """

    title_with_index = get_recommendations("ishyablockprintedkurta(setof2)")

    title_with_index = get_recommendations("pambapants")
    skulist = title_with_index.tolist()
    return skulist


def title_list():
    """
    function to get titles of store key unit values
    """

    titlelist = []
    skulist = get_sku()
    for i in skulist:
        titlefromsku = final_product[final_product["Variant SKU"] == i][["Title"]]
        titlelist.append(titlefromsku)
    return titlelist
