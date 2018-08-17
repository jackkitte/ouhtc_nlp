# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
import heapq
import MeCab
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.matutils import corpus2dense
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib as mpl
mpl.use("Agg")

mecab = MeCab.Tagger("")
mecab.parse("")
ouhtc_list = []

with open("./data/stopwords.dump", "rb") as f:
    stopwords = pickle.load(f)

with open("./data/20180101_0815/change_control_boards_20180102.txt", "r") as f:
    sp_column_name = f.readline().split("\t")
    for line in f:
        sp_list = line.split("\t")
        try:
            ouhtc_list.append({
                sp_column_name[0]: sp_list[0],
                sp_column_name[5]: sp_list[5],
                sp_column_name[3]: sp_list[3],
                sp_column_name[6]: sp_list[6],
                sp_column_name[9]: sp_list[9],
                sp_column_name[13]: sp_list[13],
            })
        except:
            pass

only_customer_list = [dic for dic in ouhtc_list if dic.get("customer_name")]

with open("./data/20180101_0815/ouhtc_list.dump", "wb") as f:
    pickle.dump(ouhtc_list, f)
with open("./data/20180101_0815/only_customer_list.dump", "wb") as f:
    pickle.dump(only_customer_list, f)

word_2d_list = []
for dic in only_customer_list:
    content = re.sub(r"\n", "", dic["content"])
    parse_text = mecab.parse(content)
    parse_list = parse_text.replace("\t", ",").split("\n")[:-2]
    word_list = []
    for values in parse_list:
        values_list = values.split(",")
        if values_list[0] in stopwords:
            continue
        elif re.match(r"[-]?\d.*", values_list[0]):
            continue
        elif values_list[1] == "名詞":
            word_list.append(values_list[0])
    word_2d_list.append(word_list)

with open("./data/20180101_0815/word_2d_list.dump", "wb") as f:
    pickle.dump(word_2d_list, f)

flatten_list = [flatten for inner in word_2d_list for flatten in inner]
dict_2d = Dictionary(word_2d_list)
dict_1d = Dictionary([flatten_list])

with open("./data/20180101_0815/dict_2d.dump", "wb") as f:
    pickle.dump(dict_2d, f)
with open("./data/20180101_0815/dict_1d.dump", "wb") as f:
    pickle.dump(dict_1d, f)

bow_2d = list(map(dict_2d.doc2bow, word_2d_list))
bow_1d = dict_1d.doc2bow(flatten_list)
word2freq = {}
for id_, freq in bow_1d:
    word2freq[dict_2d[id_]] = freq

with open("./data/20180101_0815/word2freq.dump", "wb") as f:
    pickle.dump(word2freq, f)

tf_idf_model = TfidfModel(bow_2d)
tf_idf_corpus = tf_idf_model[bow_2d]
tf_idf_dense = np.empty((0, len(dict_2d)))

for tf_idf in tf_idf_corpus:
    dense = corpus2dense([tf_idf], len(dict_2d)).T[0]
    tf_idf_dense = np.append(tf_idf_dense, dense[np.newaxis, :], axis=0)

with open("./data/20180101_0815/tf_idf_corpus.dump", "wb") as f:
    pickle.dump(tf_idf_corpus, f)
with open("./data/20180101_0815/tf_idf_dense.dump", "wb") as f:
    pickle.dump(tf_idf_dense, f)

lsi_model = LsiModel(tf_idf_corpus, id2word=dict_2d, num_topics=300)
lsi_corpus = lsi_model[tf_idf_corpus]
lsi_model.save("./data/20180101_0815/lsi_topics300.model")
lsi_dense = np.empty((0, len(lsi_corpus[0])))
for lsi in lsi_corpus:
    dense = corpus2dense([lsi], len(lsi_corpus[0])).T[0]
    lsi_dense = np.append(lsi_dense, dense[np.newaxis, :], axis=0)

with open("./data/20180101_0815/lsi_corpus.dump", "wb") as f:
    pickle.dump(lsi_corpus, f)
with open("./data/20180101_0815/lsi_dense.dump", "wb") as f:
    pickle.dump(lsi_dense, f)
 
kmeans = KMeans(n_clusters=10, random_state=0).fit(lsi_dense)
labels = kmeans.labels_
labels_corpus = {}
for label, lsi in zip(labels, lsi_corpus):
    if label not in labels_corpus.keys():
        labels_corpus[label] = {}
    lsi_abs = [(x, abs(y)) for x, y in lsi]
    for id_, value in heapq.nlargest(4, lsi_abs, key=lambda x: x[1]):
        if id_ not in labels_corpus[label].keys():
            labels_corpus[label][id_] = 1
        else:
            labels_corpus[label][id_] += 1

labels_topics = {}
for label, topic_freq in labels_corpus.items():
    if label not in labels_topics.keys():
        labels_topics[label] = {}
    for id_, _ in heapq.nlargest(4, topic_freq.items(), key=lambda x: x[1]):
        for word, val in lsi_model.show_topic(id_, topn=100):
            if word not in labels_topics[label].keys():
                labels_topics[label][word] = abs(val)
            else:
                max_val = max(labels_topics[label][word], abs(val))
                labels_topics[label][word] = abs(max_val)

with open("./data/20180101_0815/labels_corpus.dump", "wb") as f:
    pickle.dump(labels_corpus, f)
with open("./data/20180101_0815/labels_topics.dump", "wb") as f:
    pickle.dump(labels_topics, f)

font_path="/temp/IPAexfont00301/ipaexg.ttf"
for label, corpus in labels_topics.items():
    wordcloud = WordCloud(background_color="white", font_path=font_path, width=900, height=500).generate_from_frequencies(corpus)
    wordcloud.to_file("./images/20180101_0815/wordcloud_lsi_freq_label{}.png".format(label))