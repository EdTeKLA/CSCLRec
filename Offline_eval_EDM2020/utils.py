import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
import nltk
import spacy
import numpy as np
import random
from nltk.tokenize import TweetTokenizer
import collections
from collections import defaultdict
import math
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.spatial.distance import cosine

def lemmatization_filter(nlp, texts, allowed_postags):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def pre_lemmatize(data_words):
    lmtzr = nltk.WordNetLemmatizer()
    output = []
    for sent in data_words:
        # print(tokenized_phrases)
        output.append([lmtzr.lemmatize(token) for token in sent])
    return output

def preprocess_raw(all_note_contents):
    """
    Not filtering stop words
    """

    data_words = []
    note_ids = []  # ids of all non-empty posts
    for i, row in all_note_contents.iterrows():
        if not pd.isnull(row['Contents']):
            data_words.append(gensim.utils.simple_preprocess(str(row['Contents'])))
            note_ids.append(row['NoteID'])
    
    # Lemmatize
    data_words = pre_lemmatize(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]  # doc2bow returns BoW format of a list

    return data_words, note_ids, id2word, corpus

def preprocess(all_note_contents, min_count, allowed_pos_tags, stopwords, token_option):
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    data_words = []
    note_ids = []  # ids of all non-empty posts
    for i, row in all_note_contents.iterrows():
        if not pd.isnull(row['Contents']):
            data_words.append(gensim.utils.simple_preprocess(str(row['Contents'])))
            note_ids.append(row['NoteID'])
    # Lemmatize
    data_words = pre_lemmatize(data_words)

    # we extract tokens, and phrases separately
    # Tokens
    # Remove Stop Words
    tokens = [[word for word in doc if word not in stopwords]
              for doc in data_words]

    if token_option == 'tokens_only':
        frequency = defaultdict(int)
        for post in tokens:
            for token in post:
                frequency[token] += 1
        tokens_hifreq = [[token for token in post if frequency[token] > 3]
                         for post in tokens]
        data_words = lemmatization_filter(nlp, tokens_hifreq, allowed_postags=allowed_pos_tags)
    elif token_option == 'tokens_phrases':
        # Phrases
        # Prepare into sentences seperated by comma/semicolumn/column/?/!/'/"/
        iterator = all_note_contents.iterrows()
        punkt_tknzr = nltk.data.load('tokenizers/punkt/english.pickle')
        tweet_tknzr = TweetTokenizer()
        dic_tok = {post['NoteID']: [tweet_tknzr.tokenize(sent)[:-1]
                                    for sent in punkt_tknzr.tokenize(post['Contents']
                                                                     .replace(',', '.').replace('?', '.')
                                                                     .replace('/', '.').replace('-', '.')
                                                                     .replace(')', '.').replace('(', '.')
                                                                     .replace(';', '.').replace('!', '.')
                                                                     .replace(':', '.').replace('"', '.'))]
                   for i, post in iterator if pd.notnull(post['Contents'])}
        phrases_list_overall = []
        note_ids = []
        for note_id, post in dic_tok.items():
            for sent in post:
                phrases_list_overall.append(sent)
            note_ids.append(note_id)
        # feed into phraser to build the bigram and trigram models, out of all texts
        # https://stackoverflow.com/a/35748858
        # accept as a phrase if (count(a, b) - min_count) * N / (count(a) * count(b)) > threshold
        # higher threshold fewer phrases.
        bigram = gensim.models.Phrases(phrases_list_overall, min_count=5, threshold=20)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram = gensim.models.Phrases(bigram_mod[phrases_list_overall], min_count=5, threshold=20)
        trigram_mod = gensim.models.phrases.Phraser(trigram)  # Finished training of bigram/trigram parser
        # Form Trigrams
        trigrams = [trigram_mod[doc] for doc in tokens]
        frequency = defaultdict(int)
        for sent in trigrams:
            for phrase in sent:
                frequency[phrase] += 1
        trigrams_hifreq = [[phrase for phrase in sent if frequency[phrase] > min_count]
                           for sent in trigrams]
        if allowed_pos_tags: # if has specified certain postags allowed
            data_words = lemmatization_filter(nlp, trigrams_hifreq, allowed_postags=allowed_pos_tags)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]  # doc2bow returns BoW format of a list

    return data_words, note_ids, id2word, corpus

def extract_vocab(all_note_contents, min_count, allowed_pos_tags, stopwords):
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    data_words = []
    note_ids = []  # ids of all non-empty posts
    for i, row in all_note_contents.iterrows():
        if not pd.isnull(row['Contents']):
            data_words.append(gensim.utils.simple_preprocess(str(row['Contents'])))
            note_ids.append(row['NoteID'])
    # Lemmatize
    data_words = pre_lemmatize(data_words)

    # we extract tokens, and phrases separately
    # Tokens
    # Remove Stop Words
    tokens = [[word for word in doc if word not in stopwords]
              for doc in data_words]
    # Phrases
    # Prepare into sentences seperated by comma/semicolumn/column/?/!/'/"/
    iterator = all_note_contents.iterrows()
    punkt_tknzr = nltk.data.load('tokenizers/punkt/english.pickle')
    tweet_tknzr = TweetTokenizer()
    dic_tok = {post['NoteID']: [tweet_tknzr.tokenize(sent)[:-1]
                                for sent in punkt_tknzr.tokenize(post['Contents']
                                                                 .replace(',', '.').replace('?', '.')
                                                                 .replace('/', '.').replace('-', '.')
                                                                 .replace(')', '.').replace('(', '.')
                                                                 .replace(';', '.').replace('!', '.')
                                                                 .replace(':', '.').replace('"', '.'))]
               for i, post in iterator if pd.notnull(post['Contents'])}
    phrases_list_overall = []
    note_ids = []
    for note_id, post in dic_tok.items():
        for sent in post:
            phrases_list_overall.append(sent)
        note_ids.append(note_id)
    # feed into phraser to build the bigram and trigram models, out of all texts
    # https://stackoverflow.com/a/35748858
    # accept as a phrase if (count(a, b) - min_count) * N / (count(a) * count(b)) > threshold
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(phrases_list_overall, min_count=3, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram_mod[phrases_list_overall], min_count=3, threshold=10)
    trigram_mod = gensim.models.phrases.Phraser(trigram)  # Finished training of bigram/trigram parser
    # Form Trigrams
    trigrams = [trigram_mod[doc] for doc in tokens]
    frequency = defaultdict(int)
    for sent in trigrams:
        for phrase in sent:
            frequency[phrase] += 1
    trigrams_hifreq = [[phrase for phrase in sent if frequency[phrase] > min_count]
                       for sent in trigrams]
    if allowed_pos_tags: # if has specified certain postags allowed
        data_words = lemmatization_filter(nlp, trigrams_hifreq, allowed_postags=allowed_pos_tags)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]  # doc2bow returns BoW format of a list

    return data_words, note_ids, id2word, corpus

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def leaders(xs, top=10):
    """
    Returns the top-[top] frequent items in [xs]
    """
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]


def topn_from_dict(dic, n=10):
    """
    Returns the top-[n] highest valued items from [dic]
    """
    return sorted(dic.items(), key=lambda kv: kv[1])[::-1][:n]

def get_popularity_rank(eval_nids, popularities, n=10):
    """
    Returns the top-[n] highest valued items from [dic]
    """
    evals = {nid:value for nid,value in popularities.items() if nid in eval_nids}

    sorted_evals = sorted(evals.items(), key=lambda kv: kv[1])[::-1][:n]

    return set([e[0] for e in sorted_evals])

def leastn_from_dict(dic, n=10):
    """
    Returns the top-[n] lowest valued items from [dic]
    """
    return sorted(dic.items(), key=lambda kv: kv[1])[:n]

def all_sorted(xs):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])


def threshold_leaders(xs, thres=2):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    lst = sorted(counts.items(), reverse=True, key=lambda tup: tup[1])
    return [e for e in lst if e[1] >= thres]


def tf_dict(word_list):
    tfdict = {}
    for p in word_list:
        if p not in tfdict:
            tfdict[p] = word_list.count(p) / float(len(word_list))
    return tfdict


def idf_dict(all_posts_word_lists):
    num_posts = len(all_posts_word_lists)
    bow = set()
    for post in all_posts_word_lists:
        for word in post:
            bow.add(word)

    idfdict = {}
    post_appear_dict = {w: [] for w in bow}
    for word in bow:
        post_idx = 0
        for post_word_list in all_posts_word_lists:
            if word in post_word_list:
                post_appear_dict[word].append(post_idx)
            post_idx += 1
        idfdict[word] = math.log10(num_posts / float(len(post_appear_dict[word])))

    return (idfdict, post_appear_dict)


def tfidf(data_words):
    output = []
    idfdict, post_appear_dict = idf_dict(data_words)
    tfdicts = []
    for post_word_list in data_words:
        tfidfs_of_post = {}
        tfdict = tf_dict(post_word_list)
        for word, tf_val in tfdict.items():
            tfidfs_of_post[word] = tf_val * idfdict[word]
        tfdicts.append(tfdict)
        output.append(tfidfs_of_post)
    return output, tfdicts, post_appear_dict

def get_top_tfidfs(tfidf_m, num=3):
    """
    Return the keywords extracted from the tfidf matrix
    num: top-n tfidf scores
    """
    sorted_x = sorted(tfidf_m.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_x)
    outlist = list(sorted_dict)

    return outlist[-int(num):]

def get_topic_words(tfidf_m, num=3):
    """
    Return the keywords extracted from the tfidf matrix
    num: top-n tfidf scores
    """
    sorted_x = sorted(tfidf_m.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_x)
    # print(sorted_dict)
    outlist = list(sorted_dict)
    return outlist[-num:][::1]

#
#  Copyright 2016-2018 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# euclidean distance between two vectors
def l2_dist(v1, v2):
    sum = 0.0
    if len(v1) == len(v2):
        for i in range(len(v1)):
            delta = v1[i] - v2[i]
            sum += delta * delta
        return math.sqrt(sum)

def l2_sim(v1, v2):
    sum = 0.0
    if len(v1) == len(v2):
        for i in range(len(v1)):
            delta = v1[i] - v2[i]
            sum += delta * delta
        return 1/math.sqrt(sum)

def cosine_sim(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

def cosine_dist(v1, v2):
    return cosine(v1, v2)

#https://stackoverflow.com/a/37220870
from math import pi, acos
def similarity(x, y):
    return sum(x[k] * y[k] for k in x if k in y) / sum(v**2 for v in x)**.5 / sum(v**2 for v in y)**.5
#https://stackoverflow.com/a/37220870
def distance_metric(x, y):
    return 1 - 2 * acos(similarity(x, y)) / pi

def ild(v1, v2):
    return

def under_same_thread():
    return

def structural_sim(note_id1, note_id2):
    """
    Return overall numeric structural similarity value
    """
    return

def get_random_list(the_list, k):
    if k<=len(the_list):
        return random.sample(the_list, k)
    else:
        return the_list

def is_direct_reply(hier, note_id1, note_id2):
    if note_id1 in hier[note_id2] or note_id2 in hier[note_id1]:
        return True
    else:
        return False

def is_sibling(hier, note_id1, note_id2):
    for replies in hier.values():
        
        if set([note_id1, note_id2]).issubset(set(replies)):
            return True
        
    return False

import numpy as np

'''
https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
'''
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        #return 0.0
        return 1.0

    return score / min(len(actual), k)
    
'''
https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
'''
def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

