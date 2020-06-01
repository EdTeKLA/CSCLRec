import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
# import pickle
# import seaborn as sns
# from scipy import stats
import utils
# import operator
import spacy
# import re
# from nltk.corpus import wordnet as wn   
# import sys
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from gensim.models.wrappers import FastText
from gensim.models import LsiModel
#from gensim.models.fasttext import *
#import pyemd
import logging

class Preprocessor:
    """
    An object used to read in data from files, extract interactions to dataframes 
    and pre-compute textual embeddings and similarities.
    
    When offline, it is also used to derive training and test data
    
    """
    def __init__(self, path_sample, dbname, weight_post, weight_user, start_date, verbose=0):
        self.dbname = dbname
        self.weight_post = list(weight_post.keys())
        self.weight_user = list(weight_user.keys())
        self.path_sample = path_sample
        self.start_date = start_date
        self.verbose = verbose

        # Read in files
        df_raw, df_like, df_link, df_reading, df_reada = self.read_file()
        self.df = self.clean_noteinfo(df_raw)
        self.df_like = self.clean_like_link(df_like)
        self.df_link = self.clean_like_link(df_link)
        self.df_reading = df_reading.loc[(df_reading.DatabaseID == dbname)]
        self.df_reada = df_reada.loc[(df_reada.DatabaseID == dbname)]

        # Extract interactions
        self.post_interactions_per_person = self.get_post_interactions()
        self.user_interactions_per_person = self.get_user_interactions()
        self.all_post_interactions = pd.concat([df for df in self.post_interactions_per_person.values()]).drop_duplicates(['NoteID']).reset_index(drop=True)
        self.all_note_contents = self.all_post_interactions[['NoteID', 'Contents','AwlCount']]
        # Empty, Deleted and Private posts are pre-filtered here
        # They will not be used to neither training(graph-building) nor recommendation 
        self.unsharable_posts = self.df.loc[(self.df['isEmpty'] == True) |
                                         (self.df['NoteDeleted'] == 1) |
                                         (self.df['Private'] == 1), 'NoteID'].values
        self.hierarchy = self.get_hierarchy()
        self.post_similarities = {}

    def read_file(self):
        #os.chdir(self.path_sample)
        df_raw = pd.read_json(self.path_sample+'/%d/noteinfo.json'%self.dbname) 
        df_like = pd.read_json(self.path_sample+'/%d/likes.json'%self.dbname)
        df_link = pd.read_json(self.path_sample+'/%d/tolink.json'%self.dbname)
        df_reading = pd.read_csv(self.path_sample+'/%d/reading.csv'%self.dbname)
        df_reada = pd.read_csv(self.path_sample+'/%d/reada.csv'%self.dbname)

        if df_link.empty:
            df_link = pd.DataFrame(columns=['DatabaseID', 'NoteContents', 'ToNoteID', 'Weeknum','LinkPersonID', 
                            'NoteAuthorPersonID','DateCreated','AuthorAccountID','VNoteID','LinkID','LinkAlert']) 
        s1, e1 = self.convert_date(df_raw, 'DateCreated')
        s2, e2 = self.convert_date(df_like, 'TimeStamp')
        s3, e3 = self.convert_date(df_link, 'DateCreated')
        s4, e4 = self.convert_date(df_reading, 'DateRead')
        s5, e5 = self.convert_date(df_reada, 'DateRead')

        start_date = self.start_date

        df_raw = self.add_week_num(df_raw, 'DateCreated', start_date)
        df_like = self.add_week_num(df_like, 'TimeStamp', start_date)
        df_link = self.add_week_num(df_link, 'DateCreated', start_date)
        df_reading = self.add_week_num(df_reading, 'DateRead', start_date)
        df_reada = self.add_week_num(df_reada, 'DateRead', start_date)

        return df_raw, df_like, df_link, df_reading, df_reada
    
    def convert_date(self, df, time_column):
        if time_column in df:
            df['dt_py'] = pd.to_datetime(df[time_column])
            df['year'] = df['dt_py'].dt.year
        else:
            logging.debug('No such column in datetime format')
        

        end_date = df['dt_py'].max()
        start_date = df['dt_py'].min()
        
        if self.verbose:
            cross_year = (len(df['year'].unique())>1)
            if cross_year:
                logging.debug('NOTE: this course runs crossing a year')

            logging.debug(time_column)
            logging.debug(len(df))
            logging.debug("this course runs from %s to %s",start_date.strftime("%m/%d/%Y, %H:%M:%S"), 
                          end_date.strftime("%m/%d/%Y, %H:%M:%S"))
        
        return start_date, end_date


    def add_week_num(self, df, time_column, start_date):
        if time_column in df:
            df['Weeknum'] = df['dt_py'].apply(lambda x: (x - start_date).days//7 + 1 if x>=start_date else 1)  
        else:
            logging.error('No such column in datetime format')
        
        return df

    def clean_noteinfo(self, df_m):
        unused_columns = ['DatabaseID', 'AttachedFileText',
                          'WordCount', 'SignoutID', 'AttachedFileFlag', 'CheckMark',
                          'SignoutTime', 'PubliclyEditableFlag', 'SharedFlag']

        for col in unused_columns:
            df_m = df_m.drop(col, axis=1)

        temp_d = []
        for index, row in df_m.iterrows():

            row['isEmpty'] = False
            try:
                soup = BeautifulSoup(row.NoteContents, 'lxml')

                text = soup.get_text()
                row['Contents'] = text.replace("\r\n", "\n").replace('\r', '\n')\
                                    .replace('\n', ' ').replace('\t', ' ')\
                                    .replace(u'\xa0', u' ').replace('   ',' ').replace('  ',' ')

                temp_d.append(row)
            except TypeError:
                row['isEmpty'] = True
                temp_d.append(row)

        df_new = pd.DataFrame(temp_d)
        df_new = df_new.drop('NoteContents', axis=1)

        return df_new.reset_index(drop=True)

    def clean_like_link(self, df):
        temp_d = []
        for index, row in df.iterrows():
            row['isEmpty'] = False
            try:
                soup = BeautifulSoup(row.NoteContents, 'lxml')
                [s.extract() for s in soup(['script', 'a'])]
                text = soup.get_text()
                row['Contents'] = text.replace("\r\n", "\n").replace('\r', '\n')\
                                    .replace('\n', ' ').replace('\t', ' ')\
                                    .replace(u'\xa0', u' ').replace('   ',' ').replace('  ',' ')
                temp_d.append(row)
            except TypeError:
                row['isEmpty'] = True
                temp_d.append(row)

        df_new = pd.DataFrame(temp_d)
        df_new = df_new.drop('NoteContents', axis=1)

        return df_new.reset_index(drop=True)



    def get_user_interactions(self):
        user_interactions_per_person = {}
        for pid in self.df.PersonID.unique():
            user_interactions_per_person[pid] = self.extract_user_interactions(pid, self.weight_user)
        return user_interactions_per_person

    def get_post_interactions(self):
        post_interactions_per_person = {}
        for pid in self.df.PersonID.unique():
            post_interactions_per_person[pid] = self.extract_post_interactions(pid, self.weight_post)
        return post_interactions_per_person

    def revisit_week(self, note_ids, note_id):
        """
        https://stackoverflow.com/a/5419576
        """
        start_at = -1
        locs = []
        while True:
            try:
                loc = note_ids.index(note_id, start_at + 1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

    def extract_post_interactions(self, pid, weight):
        # user-created original posts
        df_created = self.df.loc[self.df.PersonID == pid, ['Contents', 'NoteID', 'Weeknum','AwlCount', 'dt_py']]
        df_created['weight'] = [weight[0] for i in range(len(df_created))]

        # posts that the user liked
        df_like_out = self.df_like.loc[(self.df_like.PersonID == pid), ['Contents', 'NoteID', 'Weeknum','AwlCount', 'dt_py']]
        df_like_out['weight'] = [weight[1] for i in range(len(df_like_out))]

        # posts that the user linked to
        df_link_out = self.df_link.loc[(self.df_link.LinkPersonID == pid), ['Contents', 'ToNoteID', 'Weeknum','AwlCount', 'dt_py']]
        df_link_out.rename(columns={'ToNoteID': 'NoteID'}, inplace=True)
        df_link_out['weight'] = [weight[2] for i in range(len(df_link_out))]

        # posts that the user has replied to
        df_built_on_p = self.df.loc[
            self.df.PersonID == pid, ['isBuiltOn', 'BuildsOn', 'Weeknum', 'dt_py']]  # .reset_index(drop=True)
        built_on_indices = filter(lambda a: a != 0, df_built_on_p['BuildsOn'].values.tolist())
        df_built_on_tp = self.df.loc[self.df.NoteID.isin(built_on_indices), ['Contents', 'NoteID', 'Weeknum','AwlCount', 'dt_py']]
        for idx, row in df_built_on_tp.iterrows():
            wk = df_built_on_p.loc[df_built_on_p.BuildsOn == row.NoteID, 'Weeknum'].values[0]
            df_built_on_tp.at[idx, 'Weeknum'] = wk
        df_built_on_tp['weight'] = [weight[3] for i in range(len(df_built_on_tp))]

        # readings
        df_reading_p = self.df_reading.loc[self.df_reading.ViewerPersonID == pid, ['NoteID', 'Weeknum','AwlCount', 'dt_py']]
        df_reading_tp = self.df.loc[
            self.df.NoteID.isin(df_reading_p.NoteID.values.tolist()), ['Contents', 'NoteID', 'Weeknum','AwlCount', 'dt_py']]
        for idx, row in df_reading_tp.iterrows():
            wk = df_reading_p.loc[df_reading_p.NoteID == row.NoteID, 'Weeknum'].values[0]
            df_reading_tp.at[idx, 'Weeknum'] = wk
        df_reading_tp['weight'] = [weight[5] for i in range(len(df_reading_tp))]

        # anonymous readings
        df_reada_p = self.df_reada.loc[self.df_reada.ViewerPersonID == pid, ['NoteID', 'Weeknum','AwlCount', 'dt_py']]
        # print(len(df_reada_p))
        df_reada_tp = self.df.loc[
            self.df.NoteID.isin(df_reada_p.NoteID.values.tolist()), ['Contents', 'NoteID', 'Weeknum','AwlCount', 'dt_py']]
        # print(len(df_reada_tp))
        for idx, row in df_reada_tp.iterrows():
            wk = df_reada_p.loc[df_reada_p.NoteID == row.NoteID, 'Weeknum'].values[0]
            df_reada_tp.at[idx, 'Weeknum'] = wk
        df_reada_tp['weight'] = [weight[6] for i in range(len(df_reada_tp))]

        interactions = pd.concat([df_created, df_like_out, df_link_out,
                                  df_built_on_tp, df_reading_tp, df_reada_tp], sort=False)

        # Re-visit
        df_views = interactions.loc[interactions['weight'].isin([weight[5], weight[6]])].reset_index(drop=True)
        noteids = list(df_views.NoteID.values)
        weeknums = df_views.Weeknum.values
        contents = df_views.Contents.values
        dts = df_views.dt_py.values
        awls = df_views.AwlCount.values
        revisit_weight = weight[4]
        list_revisits = []
        for nid in df_views.NoteID.unique():
            revisit_indices = self.revisit_week(noteids, nid)[1:]
            for i in revisit_indices:
                week = weeknums[i]
                dt = dts[i]
                text_content = contents[i]
                awl_count = awls[i]
                list_revisits.append([text_content, nid, week, revisit_weight, awl_count, dt])
        df_revisit = pd.DataFrame(list_revisits, columns=['Contents', 'NoteID', 'Weeknum', 'weight','AwlCount','dt_py'])
        interactions = pd.concat([interactions, df_revisit], sort=False)
        return interactions

    def extract_user_interactions(self, pid, weight):
        # posts that the user liked
        df_like_out = self.df_like.loc[(self.df_like.PersonID == pid), ['PersonID', 
                                                                        'NoteAuthorID', 
                                                                        'Weeknum']]
        df_like_out.rename(columns={'PersonID': 'PID_OUT', 'NoteAuthorID': 'PID_IN'}, inplace=True)
        df_like_out['weight'] = [weight[0] for i in range(len(df_like_out))]

        # posts that the user linked to
        df_link_out = self.df_link.loc[(self.df_link.LinkPersonID == pid), ['LinkPersonID', 
                                                                            'NoteAuthorPersonID', 
                                                                            'Weeknum']]
        df_link_out.rename(columns={'LinkPersonID': 'PID_OUT', 'NoteAuthorPersonID': 'PID_IN'}, inplace=True)
        df_link_out['weight'] = [weight[1] for i in range(len(df_link_out))]

        # posts that the user has replied to
        df_built_on_p = self.df.loc[self.df.PersonID == pid, ['isBuiltOn', 'BuildsOn', 
                                                              'Weeknum', 'PersonID']]
        df_built_on_p.rename(columns={'PersonID': 'PID_OUT'}, inplace=True)
        built_on_indices = filter(lambda a: a != 0, df_built_on_p['BuildsOn'].values.tolist())
        df_built_on_tp = self.df.loc[self.df.NoteID.isin(built_on_indices), ['NoteID', 'PersonID']]
        df_built_on_tp.rename(columns={'PersonID': 'PID_IN'}, inplace=True)
        for idx, row in df_built_on_tp.iterrows():
            wk, p_out = df_built_on_p.loc[df_built_on_p.BuildsOn == row.NoteID, ['Weeknum', 'PID_OUT']].values[0]
            df_built_on_tp.at[idx, 'Weeknum'] = wk
            df_built_on_tp.at[idx, 'PID_OUT'] = p_out
        df_built_on_tp = df_built_on_tp.drop('NoteID', axis=1)
        df_built_on_tp['weight'] = [weight[2] for i in range(len(df_built_on_tp))]

        # views
        df_reading_tp = self.df_reading.loc[
            self.df_reading.ViewerPersonID == pid, ['ViewerPersonID', 'AuthorPersonID', 'Weeknum']]
        df_reading_tp.rename(columns={'ViewerPersonID': 'PID_OUT', 'AuthorPersonID': 'PID_IN'}, inplace=True)
        df_reading_tp['weight'] = [weight[3] for i in range(len(df_reading_tp))]

        # anonymous views
        df_reada_tp = self.df_reada.loc[
            self.df_reada.ViewerPersonID == pid, ['ViewerPersonID', 'AuthorPersonID', 'Weeknum']]
        df_reada_tp.rename(columns={'ViewerPersonID': 'PID_OUT', 'AuthorPersonID': 'PID_IN'}, inplace=True)
        df_reada_tp['weight'] = [weight[4] for i in range(len(df_reada_tp))]

        interactions = pd.concat([df_like_out, df_link_out,
                                  df_built_on_tp, df_reading_tp, df_reada_tp], sort=False)

        return interactions

    def preview_interaction_distribution(self):
        user_interactions_all = pd.concat(list(self.user_interactions_per_person.values()))
        post_interactions_all = pd.concat(list(self.post_interactions_per_person.values()))
        for pid, inter in self.user_interactions_per_person.items():
            print(pid)
            print(inter.groupby('Weeknum').count())
        print(user_interactions_all.groupby('Weeknum').count())
        print(post_interactions_all.groupby('Weeknum').count())

    def plot_dist(self):
        sns_plot = sns.distplot(self.df.Weeknum)
        #sns_plot.savefig("dist.png")


    def partition_data(self, cutoff_week):

        train_post_inter_per_person = {}
        test_post_inter_per_person = {}
        for pid, inter in self.post_interactions_per_person.items():
            train_post_inter_per_person[pid] = inter.loc[inter.Weeknum <= cutoff_week]
            test_post_inter_per_person[pid] = inter.loc[inter.Weeknum > cutoff_week]

        train_user_inter_per_person = {}
        test_user_inter_per_person = {}
        for pid, inter in self.user_interactions_per_person.items():
            train_user_inter_per_person[pid] = inter.loc[inter.Weeknum <= cutoff_week]
            test_user_inter_per_person[pid] = inter.loc[inter.Weeknum > cutoff_week]

        return (train_post_inter_per_person, test_post_inter_per_person,
                train_user_inter_per_person, test_user_inter_per_person)

    def get_unsharable_posts(self):
        return self.unsharable_posts

    def get_hierarchy(self):
        hierarchy = {k: [] for k in self.df.NoteID.unique()}
        for idx, p in self.df.iterrows():
            if p.BuildsOn in hierarchy.keys():
                hierarchy[p.BuildsOn].append(p.NoteID)
        return hierarchy

    def display_hierarchy(self):
        hierarchy = self.get_hierarchy()
        
        topic_nids = []
        for idx,p in self.df.iterrows():
            if p.BuildsOn==0:
                topic_nids.append(p.NoteID)
                
        hierarchy[0] = topic_nids

        def print_hier(parent,level):
            if len(hierarchy[parent])==0:
                return
            else:
                for ch in hierarchy[parent]:
                    print(level*'  ',ch)
                    print_hier(ch,level+1)

        print_hier(0,0)


    def precompute_similarity(self, model_path, option, feature_size):
        all_note_contents = self.all_note_contents.Contents.to_list()
        all_note_nids = self.all_note_contents.NoteID.to_list()
        similarity_matrix = {}

        if option == 'ft_word_emd+sif':
            #'cleaned_data/ft_model_incr'
            # Load pretrained FastText embeddings
            model = FastText.load(model_path)
            logging.info('[Preprocessor] Using model: %s',str(model))

            similarity_matrix = {}
            nlp = spacy.load("en_core_web_sm")
            note_ids = all_note_contents['NoteID'].values
            contents = all_note_contents['Contents'].values
            data_words = [[token.text for token in nlp(content)] for note_id, content in zip(note_ids, contents)]
            post_tokens = pd.DataFrame(data={'NoteID':note_ids,'Tokens':data_words}).set_index('NoteID')

            sentence_list = []
            sentence_embs = {}
            for note_id, post in post_tokens.iterrows():
                word_list = []
                for word in post.values[0]:
                    word_emd = model[word]
                    word_list.append(Word(word, word_emd))
                if len(word_list) > 0:  # did we find any words (not an empty set)
                    sentence_list.append(Sentence(word_list))
                sentence_embs[note_id] = sentence_to_vec(sentence_list, feature_size)
        
            # Compute post-wise cosine similarities
            for note_id1, emb1 in sentence_embs.items():
                for note_id2, emb2 in sentence_embs.items():
                    if note_id1!=note_id2 and (note_id2, note_id1) not in similarity_matrix:
                        # apply l2-distance
                        #utils.l2_sim()
                        # apply cosine distance
                        sim = utils.cosine_sim(emb1[0], emb2[0])
                        similarity_matrix[(note_id1, note_id2)] = sim
                        similarity_matrix[(note_id2, note_id1)] = sim

            return similarity_matrix

        elif option == 'bert_word_emb+sif':
            # for BERT
            import torch
            from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
            from keras.preprocessing.sequence import pad_sequences

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=128)
            MAX_LEN = 512
            tokenized_texts_list = []
            indexed_tokens_list = []
            attention_masks = []

            for text in all_note_contents.Contents.values:
                marked_text = "[CLS] " + text + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)
                tokenized_texts_list.append(tokenized_text)
                indexed_tokens_list.append(tokenizer.convert_tokens_to_ids(tokenized_text))

            input_ids_list = pad_sequences(indexed_tokens, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
            for seq in input_ids:
                seq_mask = [int(float(i>0)) for i in seq]
                attention_masks.append(seq_mask)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(input_ids_list)
            segments_tensors = torch.tensor(attention_masks)

            # Load pre-trained model (weights)
            model = BertModel.from_pretrained('bert-base-uncased')

            # Put the model in "evaluation" mode, meaning feed-forward operation.
            model.eval()

            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)

            emb_layers = encoded_layers[-4:]
            sum_layers = torch.stack(emb_layers, dim=0).sum(dim=0) # 434*512*768
            sentence_word_embs = {}
            for i in range(len(tokenized_texts_list)):
                sentence_word_embs[nids[i]] = sum_layers[i][:len(tokenized_texts_list[i])]

            # Keep a look up dictionary [note id] --> text content
            tokenized_texts_ = {nid: tokenized_texts_list[i] for i, nid in enumerate(nids)}

            embedding_size = feature_size # Set the shape of the sentence/post embeddings
            sentence_list = []
            note_ids_lookup = []
            for note_id in nids:
                #print(note_id)
                word_list = []
                for j in range(len(sentence_word_embs[note_id])):
                    word_emb = sentence_word_embs[note_id][j]
                    # Add here if to use only keywords
                    word_text = tokenized_texts_[note_id][j] 
                    word_list.append(Word(word_text, word_emb.numpy()))
                if len(word_list) > 0: 
                    sentence_list.append(Sentence(word_list))
                    note_ids_lookup.append(note_id) # in case there are some posts of 0 length, thus not included in this

            # Encode sentences/posts with embeddigns
            sentence_embs = {}
            sentence_vectors = sentence_to_vec(sentence_list, embedding_size)  # all vectors converted together
            if len(sentence_vectors) == len(sentence_list):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[note_ids_lookup[i]] = sentence_vectors[i]

            # Compute post-wise cosine similarities
            for note_id1, emb1 in sentence_embs.items():
                for note_id2, emb2 in sentence_embs.items():
                    if note_id1!=note_id2 and (note_id2, note_id1) not in similarity_matrix:
                        # apply l2-distance
                        #utils.l2_sim()
                        # apply cosine distance
                        sim = utils.cosine_sim(emb1[0], emb2[0])
                        similarity_matrix[(note_id1, note_id2)] = sim
                        similarity_matrix[(note_id2, note_id1)] = sim

            return similarity_matrix, sentence_embs
        
        elif option=='sentence_emb':
            import tensorflow as tf
            import tensorflow_hub as hub

            embed = hub.load(model_path)

            logging.info('[Preprocessor] using model: universal-sentence-encoder-1')
            sentence_embs = {}
            sentence_vectors = embed(all_note_contents)
            if len(sentence_vectors) == len(all_note_contents):
                for i in range(len(sentence_vectors)):
                    # map: note_id -> vector
                    sentence_embs[all_note_nids[i]] = sentence_vectors[i].numpy()

            #corr = np.inner(sentence_vectors, sentence_vectors)
            #cosine_similarities = tf.reduce_sum(tf.multiply(sentence_vectors, sentence_vectors), axis=1)
            #clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
            #sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            #print(sim_scores)
            #for i, sims in enumerate(sim_scores):
            #    for j, sim in enumerate(sims):
            ##        note_id1 = all_note_nids[i]
            #        note_id2 = all_note_nids[j]
            #        if not note_id1==note_id2:
            #            similarity_matrix[(note_id1, note_id2)] = sim 

            # Compute post-wise cosine similarities
            for note_id1, emb1 in sentence_embs.items():
                for note_id2, emb2 in sentence_embs.items():
                    if note_id1!=note_id2 and (note_id2, note_id1) not in similarity_matrix:
                        # apply l2-distance
                        #utils.l2_sim()
                        # apply cosine distance
                        sim = utils.cosine_sim(emb1, emb2)
                        similarity_matrix[(note_id1, note_id2)] = sim
                        similarity_matrix[(note_id2, note_id1)] = sim

            return similarity_matrix, sentence_embs

        elif option=='tfidf+lsi':
            logging.info('[Preprocessor] using TFIDF vectors, LSI for dimension reduction')
            data_words, note_ids, id2word, corpus = utils.preprocess(self.all_note_contents, 10, ['NOUN','VERB'], STOP_WORDS, 'tokens_phrases')
            #self.post_bows = pd.DataFrame(data={'NoteID':note_ids,'BoW':data_words}).set_index('NoteID')
            logging.debug('[Preprocessor] - %d non-empty posts', len(corpus))
            logging.debug('[Preprocessor] - %s extracted %d tokens/phrases' , 'tokens_phrases', len(id2word))
            tfidf_matrix, tf_dicts, post_appear_dict = utils.tfidf(data_words)

            word2id = {v: k for k, v in id2word.items()}
            tfidf_corpus = [[(word2id[pair[0]], pair[1]) for pair in post.items()] for post in tfidf_matrix]


            model = LsiModel(tfidf_corpus, num_topics=feature_size, id2word=id2word)

            sentence_embs = {}
            for i, post_tfidf in enumerate(tfidf_corpus):
                note_id = note_ids[i]
                if not note_id in sentence_embs:
                    post_repr = model[post_tfidf]
                    #print(post_repr)
                    #print(i)
                    sentence_embs[note_id] = np.array([p[1] for p in post_repr if len(post_repr) == feature_size])

            # Compute post-wise cosine similarities
            for note_id1, emb1 in sentence_embs.items():
                for note_id2, emb2 in sentence_embs.items():
                    if note_id1!=note_id2 and (note_id2, note_id1) not in similarity_matrix:
                        if len(emb1) and len(emb2):
                            # apply l2-distance
                            #utils.l2_sim()
                            # apply cosine distance
                            sim = utils.cosine_sim(emb1, emb2)
                            similarity_matrix[(note_id1, note_id2)] = sim
                            similarity_matrix[(note_id2, note_id1)] = sim

            return similarity_matrix, sentence_embs
