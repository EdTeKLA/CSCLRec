import os
import sys
import pickle
import numpy as np
import pickle
import os
import pandas as pd
import utils
from gensim.models import LsiModel
from Preprocessor import Preprocessor
from scipy import spatial
import operator
import logging


class CBF():
    """
        The class contains all experiments of semantic-aware content-based recommender algorithms.
    """
    def __init__(self, cutoff_weight, option):
        self.option = option
        self.user_profiles = {}
        self.items = {}
        self.feature_size = 0
        self.cutoff_weight = cutoff_weight
        self.post_interactions = None
        self.has_post_embedded = None
        self.all_note_contents = None
        self.post_bows = None
        self.top_ratio = None

        if option == 'ft_word_emb+sif' or option == 'keyword+ft_word_emb+sif':
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
            # Load pretrained FastText embeddings
            self.model = FastText.load('cleaned_data/ft_model_incr')
            logging.info('[CBF] using model: %s',str(self.model))

        elif option == 'bert_word_emb+sif' or option == 'keyword_bert_word_emb+sif':
            # Load pre-trained model (weights)
            self.model = BertModel.from_pretrained('bert-base-uncased')
            logging.info('[CBF] using model: %s',str(self.model))

        elif option == 'sentence_emb':
            self.model = hub.load('cleaned_data/1/')
            logging.info('[CBF] using model: universal-sentence-encoder-1')
        


    def load_precomputed_emb(self, embs, post_interactions):
        self.items = embs
        self.feature_size = 128#300
        self.post_interactions = post_interactions
        self.has_post_embedded = True



    def fit(self, post_interactions, min_count, allowed_pos, stopwords, 
        preprocess_option, feature_size, top_ratio):
        self.post_interactions = post_interactions
        self.feature_size = feature_size
        self.top_ratio = top_ratio

        self.all_note_contents = pd.concat([df[['NoteID', 'Contents']] for df in post_interactions.values()]).drop_duplicates(['NoteID']) \
            .reset_index(drop=True)

        logging.debug('[CBF] received %d interactions for the active user', len(self.all_note_contents))
        self.has_post_embedded = self.embed_posts(min_count, allowed_pos, stopwords, preprocess_option)


    def embed_posts(self, min_count, allowed_pos, stopwords, preprocess_option):
            
        if self.option == 'tfidf+lsi':
            logging.info('CBF - Using TFIDF vectors, LSI for dimension reduction')
            data_words, note_ids, id2word, corpus = utils.preprocess(self.all_note_contents, min_count,
                                                                 allowed_pos, stopwords, preprocess_option)
            #self.post_bows = pd.DataFrame(data={'NoteID':note_ids,'BoW':data_words}).set_index('NoteID')
            logging.debug('[CBF] - %d non-empty posts', len(corpus))
            logging.debug('[CBF] - %s extracted %d tokens/phrases' , preprocess_option, len(id2word))
        
            tfidf_matrix, tf_dicts, post_appear_dict = utils.tfidf(data_words)
            word2id = {v: k for k, v in id2word.items()}
            tfidf_corpus = [[(word2id[pair[0]], pair[1]) for pair in post.items()] for post in tfidf_matrix]
            model = LsiModel(tfidf_corpus, num_topics=self.feature_size, id2word=id2word)

            for i, post_tfidf in enumerate(tfidf_corpus):
                note_id = note_ids[i]
                if not note_id in self.items:
                    post_repr = model[post_tfidf]
                    self.items[note_id] = [p[1] for p in post_repr if len(post_repr) == self.feature_size]
            self.model = model
            return True

        elif self.option == 'tfidf+keywords+lsi':
            logging.info('CBF - Using TFIDF vectors on only 1/3 keywords of each post, LSI for dimension reduction')
            data_words, note_ids, id2word, corpus = utils.preprocess(self.all_note_contents, min_count,
                                                                 allowed_pos, stopwords, preprocess_option)
            #self.post_bows = pd.DataFrame(data={'NoteID':note_ids,'BoW':data_words}).set_index('NoteID')
            print('CBF - %d non-empty posts'%len(corpus))
            print('CBF - %s BoW extracted %d tokens/phrases'%(preprocess_option, len(id2word)))

            tfidf_matrix, tf_dicts, post_appear_dict = utils.tfidf(data_words)
            keywords = {i: utils.get_top_tfidfs(tfidf_matrix[i], len(tfidf_matrix[i]) // 3) # TODO:  have over-threshold phrases as the keyword
                        for i, m in enumerate(tfidf_matrix)}
            word2id = {v: k for k, v in id2word.items()}
            tfidf_corpus = [[(word2id[pair[0]], pair[1]) for pair in post.items()] for post in keywords]
            model = LsiModel(tfidf_corpus, num_topics=self.feature_size, id2word=id2word)

            for i, post_tfidf in enumerate(tfidf_corpus):
                note_id = note_ids[i]
                self.items[note_id] = model[post_tfidf]
            self.model = model
            return True

        elif self.option == 'KCB':
            from gensim import models
            data_words, note_ids, id2word, corpus = utils.preprocess(self.all_note_contents, min_count,
                                                                        allowed_pos, stopwords, preprocess_option)
            tfidf = models.TfidfModel(corpus) 
            corpus_tfidf = tfidf[corpus]
            corpus_tfidf_lst = []
            for doc in corpus_tfidf:
                doc.sort(key=operator.itemgetter(1))
                doc = doc[-len(doc)//self.top_ratio:]
                corpus_tfidf_lst.append(doc)
            # print('kt',corpus_tfidf_lst)
            lsi_model = models.LsiModel(corpus_tfidf_lst, id2word=id2word, num_topics=self.feature_size)  # initialize an LSI transformation
            corpus_lsi = lsi_model[corpus_tfidf_lst]

            for i, post_repr in enumerate(corpus_lsi):
                note_id = note_ids[i]
                self.items[note_id] = [p[1] for p in post_repr if len(post_repr) == self.feature_size]
            self.model = lsi_model
            return True

        elif self.option == 'word_emb+wmd':            
            # Load pretrained FastText embeddings
            self.model = FastText.load('cleaned_data/all_notes_model')
            # print('using model:',self.model)
            # Cannot get post embeddings from word embeddings
            # It cannot stand alone as a CBF method
            return False

        elif self.option == 'keyword+word_emb+wmd':
            # Load pretrained FastText embeddings
            self.model = FastText.load('cleaned_data/all_notes_model')
            # print('using model:',self.model)
            # Cannot get post embeddings from word embeddings
            # It cannot stand alone as a CBF method
            return False
        
        elif self.option == 'keyword+ft_word_emb+sif': # Using SIF on keyword --> sentence embedding

            data_words, note_ids, id2word, corpus = utils.preprocess(self.all_note_contents, min_count,
                                                                 allowed_pos, stopwords, preprocess_option)
            self.post_bows = pd.DataFrame(data={'NoteID':note_ids,'BoW':data_words}).set_index('NoteID')
            logging.debug('CBF - %d non-empty posts', len(corpus))
            logging.debug('CBF - %s BoW extracted %d tokens/phrases' , preprocess_option, len(id2word))

            sentence_list = []
            note_ids_lookup = []
            for note_id, post in self.post_bows.iterrows():
                word_list = []
                for word in post:
                    word_emd = self.model[word]
                    word_list.append(Word(word, word_emd))
                if len(word_list) > 0:  # did we find any words (not an empty set)
                    sentence_list.append(Sentence(word_list))
                    note_ids_lookup.append(note_id) # in case there are some posts of 0 length, thus not included in this

            sentence_embs = {}
            sentence_vectors = sentence_to_vec(sentence_list, self.feature_size)  # all vectors converted together
            if len(sentence_vectors) == len(sentence_list):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[note_ids_lookup[i]] = sentence_vectors[i]
            self.items = sentence_embs

            return True

        elif self.option == 'ft_word_emb+sif': # Using SIF on whole text --> sentence embedding

            data_words, note_ids, id2word, corpus = utils.preprocess_raw(self.all_note_contents)
            self.post_bows = pd.DataFrame(data={'NoteID':note_ids,'BoW':data_words}).set_index('NoteID')

            sentence_list = []
            note_ids_lookup = []
            for note_id, post in self.post_bows.iterrows():
                word_list = []
                for word in post:
                    word_emd = self.model[word]
                    word_list.append(Word(word, word_emd))
                if len(word_list) > 0:  # did we find any words (not an empty set)
                    sentence_list.append(Sentence(word_list))
                    note_ids_lookup.append(note_id) # in case there are some posts of 0 length, thus not included in this

            sentence_embs = {}
            sentence_vectors = sentence_to_vec(sentence_list, self.feature_size)  # all vectors converted together
            if len(sentence_vectors) == len(sentence_list):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[note_ids_lookup[i]] = sentence_vectors[i]
            self.items = sentence_embs

            return True

        elif self.option == 'bert_word_emb+sif':
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)
            #nids = [nid for nid in self.all_note_contents.NoteID.values if nid not in self.items.keys()]
            note_ids = self.all_note_contents.NoteID.to_list()
            MAX_LEN = 512
            tokenized_texts_list = []
            indexed_tokens_list = []
            attention_masks = []

            for text in self.all_note_contents.Contents.values:
                marked_text = "[CLS] " + text + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)
                tokenized_texts_list.append(tokenized_text)
                indexed_tokens_list.append(tokenizer.convert_tokens_to_ids(tokenized_text))

            input_ids_list = pad_sequences(indexed_tokens_list, maxlen=MAX_LEN, 
                                           dtype="long", truncating="post", padding="post")
            for seq in input_ids_list:
                seq_mask = [int(float(i>0)) for i in seq]
                attention_masks.append(seq_mask)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(input_ids_list)
            segments_tensors = torch.tensor(attention_masks)

            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.model.eval()
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)

            emb_layers = encoded_layers[-4:]
            sum_layers = torch.stack(emb_layers, dim=0).sum(dim=0)
            sentence_word_embs = {}
            for i in range(len(tokenized_texts_list)):
                sentence_word_embs[note_ids[i]] = sum_layers[i][:len(tokenized_texts_list[i])]
            tokenized_texts_ = {nid: tokenized_texts_list[i] for i, nid in enumerate(note_ids)}

            sentence_list = []
            note_ids_lookup = []
            for note_id in note_ids:
                #print(note_id)
                word_list = []
                for j in range(len(sentence_word_embs[note_id])):
                    word_emb = sentence_word_embs[note_id][j]
                    # Add here if to use only keywords
                    word_text = tokenized_texts_[note_id][j] 
                    word_list.append(Word(word_text, word_emb.numpy()))
                if len(word_list) > 0:  # did we find any words (not an empty set)
                    sentence_list.append(Sentence(word_list))
                    note_ids_lookup.append(note_id) # in case there are some posts of 0 length, thus not included in this
                    #print('wordlist',len(word_list))

            sentence_embs = {}
            sentence_vectors = sentence_to_vec(sentence_list, self.feature_size)  # all vectors converted together
            if len(sentence_vectors) == len(sentence_list):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[note_ids_lookup[i]] = sentence_vectors[i]
            self.items = sentence_embs

            return True

        elif self.option == 'sentence_emb': 
            note_ids = self.all_note_contents.NoteID.to_list()
            all_note_contents = self.all_note_contents['Contents'].to_list()
            
            sentence_embs = {}
            sentence_vectors = self.model[all_note_contents]
            if len(sentence_vectors) == len(all_note_contents):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[note_ids_lookup[i]] = sentence_vectors[i].numpy()
            self.items = sentence_embs

            return True

        elif self.option == 'sentence_emb_precomputed': 
            return True


    def construct_user_profiles(self, profile_option, interaction_option):
        if self.has_post_embedded:
            for pid in self.post_interactions.keys():
                self.construct_user_profile(pid, profile_option, interaction_option)
                

    def construct_user_profile(self, active_user, profile_option, interaction_option):
        """
        Construct the user profile for only the [active_user]
        """
        if profile_option == 'averaging':
            user_inters = self.post_interactions[active_user]
            if interaction_option == 'explicit_only':
                user_inters = user_inters.loc[user_inters.weight <= self.cutoff_weight]

            interacted_noteids = user_inters['NoteID'].values
            # Collect all posts that user has interacted, ignore those without reprs (typically meaningless texts)
            interacted_note_reprs = [self.items[nid] for nid in interacted_noteids
                                     if nid in self.items and len(self.items[nid]) == self.feature_size]
            # print([len(p) for p in interacted_note_reprs if len(p)!=15])
            if len(interacted_note_reprs):
                self.user_profiles[active_user] =  np.mean(interacted_note_reprs, axis=0)
            else:
                logging.warning('Because user %d have not interacted with any posts, they does not have profile', active_user)
                self.user_profiles[active_user] =  np.zeros(self.feature_size)                    


    def score(self, p1, p2, similarity_measure):
        if similarity_measure == 'cosine':
            return utils.cosine_sim(np.array(p1), np.array(p2))


    def run_user(self, active_user, eval_list, similarity_measure, k):
        if self.has_post_embedded:
            profile = list(self.user_profiles[active_user])
            scores = {}
            logging.debug('CBF - %d posts available to be recommended', len(eval_list))
            for nid, item_repr in self.items.items():
                if nid in eval_list:
                    if len(item_repr):
                        scores[nid] = self.score(item_repr, profile, similarity_measure)
                    else:
                        #logging.warning('CBF - post %d does not have repr', nid) # Should only occur in KG/SG methods
                        pass
            recs_with_scores = utils.topn_from_dict(scores, k)
            logging.debug(recs_with_scores)
            return set([item[0] for item in recs_with_scores])


if __name__ == '__main__': # When in actual use
    CUTOFF_WEIGHT = 5
    K = 10
    STOP_WORDS = []
    all_interactions = pd.DataFrame() # all interactions upto the moment

    ['tfidf+lsi', 'tokens_phrases', 15, 'averaging','explicit_implicit','cosine']
    cbf = CBFCF(CUTOFF_WEIGHT, 'tfidf+lsi')

    logging.info('Prediction')
    cbf.fit(all_interactions, 5, ['NOUN', 'VERB'], STOP_WORDS, 'tokens_phrases', 15)
    cbf.construct_user_profiles('averaging','explicit_implicit')

    for pid in all_interactions.keys():
        eval_list = func_to_extract_inactive_posts_for_user(pid)
        rec_set = cbf.run_user(pid, eval_list, 'cosine', K)