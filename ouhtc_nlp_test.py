# -*- coding: utf-8 -*-

import re
import MeCab
from urllib import request
import pickle
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.matutils import corpus2dense
import numpy as np
import heapq
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from wordcloud import WordCloud

mecab = MeCab.Tagger("")
mecab.parse('')
ouhtc_dic = {}
slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
slothlib_file = request.urlopen(slothlib_path)
slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']

with open("./data/stopwords.dump", "wb") as f:
    pickle.dump(slothlib_stopwords, f)

with open("./data/change_control_boards.txt", "r") as f:
    f.readline()
    for line in f:
        sp_list = line.split("\t")
        try:
            ouhtc_dic[sp_list[0]] = sp_list[13]
        except:
            pass

with open("./data/ouhtc_dic.dump", "wb") as f:
    pickle.dump(ouhtc_dic, f)

word_2d_list = []
for id_, content in ouhtc_dic.items():
    content = re.sub(r'\n', '', content)
    parse_text = mecab.parse(content)
    parse_list = parse_text.replace("\t", ",").split("\n")[:-2]
    word_list = []
    for values in parse_list:
        values_list = values.split(",")
        if values_list[0] in slothlib_stopwords:
            continue
        elif re.match(r'[-]?\d.*', values_list[0]):
            continue
        elif values_list[1] == "名詞":
            word_list.append(values_list[0])
    word_2d_list.append(word_list)

with open("./data/word_2d_list.dump", "wb") as f:
    pickle.dump(word_2d_list, f)

flatten_list = [flatten for inner in word_2d_list for flatten in inner]
with open("./data/flatten_list.dump", "wb") as f:
    pickle.dump(flatten_list, f)

dct_2d = Dictionary(word_2d_list)
dct_1d = Dictionary([flatten_list])

with open("./data/dict_2d.dump", "wb") as f:
    pickle.dump(dct_2d, f)

with open("./data/dict_1d.dump", "wb") as f:
    pickle.dump(dct_1d, f)

bow_1d = dct_2d.doc2bow(flatten_list)
bow_2d = list(map(dct_2d.doc2bow, word_2d_list))
token2freq_2d = {}
token2freq_1d = {}
bow_1d_id2word = {}
for id_, freq in dct_2d.dfs.items():
    token2freq_2d[dct_2d[id_]] = freq

for id_, freq in dct_1d.dfs.items():
    token2freq_1d[dct_1d[id_]] = freq

for id_, freq in bow_1d:
    bow_1d_id2word[dct_2d[id_]] = freq

tf_idf_model = TfidfModel(bow_2d)
tf_idf_corpus = tf_idf_model[bow_2d]
tf_idf_dens = np.empty((0, len(dct_2d)))

for tf_idf in tf_idf_corpus:
    dense = corpus2dense([tf_idf], len(dct_2d)).T[0]
    tf_idf_dens = np.append(tf_idf_dens, dense[np.newaxis, :], axis=0)

lsi_model = LsiModel(tf_idf_corpus, id2word=dct_2d, num_topics=200)
lsi_corpus = lsi_model[tf_idf_corpus]
lsi_model.save('./data/lsi_topics200.model')
lsi_dense = np.empty((0, len(lsi_corpus[0])))

for lsi in lsi_corpus:
    dense = corpus2dense([lsi], len(lsi_corpus[0])).T[0]
    lsi_dense = np.append(lsi_dense, dense[np.newaxis, :], axis=0)

index = np.where(tf_idf_dens != tf_idf_dens)
for i, j in zip(index[0], index[1]):
    tf_idf_dens[i][j] = 0.
#kmeans = KMeans(n_clusters=3, random_state=0).fit(tf_idf_dens)
kmeans = KMeans(n_clusters=3, random_state=0).fit(lsi_dense)
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

with open("./data/token2freq_2d.dump", "wb") as f:
    pickle.dump(token2freq_2d, f)

with open("./data/token2freq_1d.dump", "wb") as f:
    pickle.dump(token2freq_1d, f)

with open("./data/bow_1d_id2word.dump", "wb") as f:
    pickle.dump(bow_1d_id2word, f)

with open("./data/tf_idf_corpus.dump", "wb") as f:
    pickle.dump(tf_idf_corpus, f)

with open("./data/tf_idf_dens.dump", "wb") as f:
    pickle.dump(tf_idf_dens, f)

with open("./data/labels.dump", "wb") as f:
    pickle.dump(labels, f)

with open("./data/labels_corpus.dump", "wb") as f:
    pickle.dump(labels_corpus, f)

with open("./data/lsi_corpus.dump", "wb") as f:
    pickle.dump(lsi_corpus, f)

with open("./data/lsi_dense.dump", "wb") as f:
    pickle.dump(lsi_dense, f)

with open("./data/labels_topics.dump", "wb") as f:
    pickle.dump(labels_topics, f)

font_path="/temp/IPAexfont00301/ipaexg.ttf"
for label, corpus in labels_topics.items():
    wordcloud = WordCloud(background_color="white", font_path=font_path, width=900, height=500).generate_from_frequencies(corpus)
    wordcloud.to_file("./images/wordcloud_lsi_label{}.png".format(label))
