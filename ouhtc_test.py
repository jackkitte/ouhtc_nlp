# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from gensim.models import LsiModel
from wordcloud import WordCloud
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

with open("./data/20180101_0815/dict_2d.dump", "rb") as f:
    dict_2d = pickle.load(f)

lsi_model = LsiModel.load("./data/20180101_0815/lsi_topics300.model")

with open("./data/20180101_0815/labels_topics.dump", "rb") as f:
    labels_topics = pickle.load(f)

labels_topic_vec = {}
for label, many_topic in labels_topics.items():
    if label not in labels_topic_vec.keys():
        labels_topic_vec[label] = []
    topic_vec_list = []
    for topic_id, weight in many_topic:
        w_vector = lsi_model.get_topics()[topic_id] * weight
        topic_vec_list.append(w_vector)
    labels_topic_vec[label] = np.average(topic_vec_list, axis=0)

with open("./data/20180101_0815/labels_topic_vec.dump", "wb") as f:
    pickle.dump(labels_topic_vec, f)

labels_words_freq = {}
for label, vec in labels_topic_vec.items():
    if label not in labels_words_freq.keys():
        labels_words_freq[label] = {}
    for id_, val in enumerate(vec):
        labels_words_freq[label][dict_2d[id_]] = abs(val)

with open("./data/20180101_0815/labels_words_freq.dump", "wb") as f:
    pickle.dump(labels_words_freq, f)

with open("./data/20180101_0815/only_customer_dic.dump", "rb") as f:
    only_cutomer_dic = pickle.load(f)

with open("./data/20180101_0815/labels_ids.dump", "rb") as f:
    labels_ids = pickle.load(f)

content_df = pd.DataFrame()
for label, id_list in labels_ids.items():
    for id_ in id_list:
        key_list = ["label"]
        val_list = [label]
        for key, val in only_cutomer_dic[id_].items():
            key_list.append(key)
            val_list.append(val)
        tmp_se = pd.Series(val_list, index=key_list, name=id_)
        content_df = content_df.append(tmp_se)

content_df.to_csv("./data/20180101_0815/content.csv")
with open("./data/20180101_0815/content_df.dump", "wb") as f:
    pickle.dump(content_df, f)


font_path="/temp/IPAexfont00301/ipaexg.ttf"
plt.figure(dpi=300)
for label, corpus in labels_words_freq.items():
    wordcloud = WordCloud(background_color="white", font_path=font_path, width=900, height=500).generate_from_frequencies(corpus)
    wordcloud_ndarray = wordcloud.to_array()
    plt.subplot(3,4,int(label)+1)
    plt.imshow(wordcloud_ndarray)
    plt.axis("off")
    plt.title("class_{}".format(label))
    with open("./images/20180101_0815/wordcloud_lsi_freq_ndarray{}.dump".format(label), "wb") as f:
        pickle.dump(wordcloud_ndarray, f)
    wordcloud.to_file("./images/20180101_0815/wordcloud_lsi_freq_label{}_improvement.png".format(label))

plt.savefig("./images/20180101_0815/wordcloud_lsi_freq.png")