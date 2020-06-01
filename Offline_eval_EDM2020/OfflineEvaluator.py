from Preprocessor import Preprocessor
import utils
from CBF import CBF
from STOPWORDS import STOP_WORDS
from CSCLRec import CSCLRec
from ContentAnalyzer import ContentAnalyzer
from PostFilter import PostFilter
from UserProfiler import UserProfiler
from PurePPR import PurePPR
from CoPPR import CoPPR
from MCF import MCF

import pickle
import logging
import operator
import sys
import os
import random
from collections import Counter
import itertools
from itertools import combinations
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import spacy
import re
import networkx as nx

class OfflineEvaluator:
    def __init__(self, weight_post_lookup, weight_user_lookup, 
                 cutoff_weight, explicit_thres, implicit_thres, START_DATE, verbose):
        self.cutoff_weight = cutoff_weight
        self.weight_post_lookup = weight_post_lookup
        self.weight_user_lookup = weight_user_lookup
        self.weight_post = list(weight_post_lookup.keys())
        self.weight_user = list(weight_user_lookup.keys())
        self.explicit_thres = explicit_thres
        self.implicit_thres = implicit_thres
        
        self.train_post_inter = None
        self.test_post_inter = None
        self.train_user_inter = None
        self.test_user_inter = None
        self.users = None
        self.unrec_noteids = None
        self.week_num = None
        self.week_stride = None
        self.gtp_criterion = None        
        self.eval_inters_per_person = {} # Dataframes of interactions of all users that are available/visible to a specific user. (dictionary of dictionary of dataframes)
        self.eval_nids_per_person = {} # Sets of note ids that are availale/visible to a specific user. (dictionary of sets)
        self.all_post_inters = None # 
        self.all_post_inters_df = None
        self.hierarchy = None
        self.precomputed_item_similarities = None
        self.precomputed_sen_embs = None
        self.precomputed_temporal_popularity = None
        self.precomputed_keywords = None
        self.preprocessor = None
        self.all_note_contents = None
        self.verbose = verbose
        self.start_date = START_DATE
        self.target_users = None

    
    def fit(self, precomputed_item_similarities, precomputed_sen_embs, precomputed_keywords, precomputed_content_graph, instructors):
        self.precomputed_item_similarities = precomputed_item_similarities # only for evaluation,
        self.precomputed_sen_embs = precomputed_sen_embs # only for evaluation, no need to have it in production
        self.precomputed_content_graph = precomputed_content_graph # can be computed once for all (e.g during every night)
        self.precomputed_keywords = precomputed_keywords
        self.target_users = set(self.users) - set(instructors)
            
    def preprocess(self, path_sample, dbnum, week_num, week_stride):
        logging.info('........................... Preprocessing Data ..................................')
        self.week_num = week_num
        self.week_stride = week_stride
        self.eval_week = range(self.week_num+1, self.week_num+1+self.week_stride)
        
        logging.info('Fitting evaluator with week splits at %d', week_num)
        
        self.preprocessor = Preprocessor(path_sample, dbnum,
                                         self.weight_post_lookup, self.weight_user_lookup, 
                                         self.start_date, verbose=self.verbose)
        
        # preprocessor.preview_interaction_distribution()
        
        # Return every interaction incurred before cutoff week as the training interaction
        # every interaction incurred after cutoff week as the testing interaction
        # post interactions are user2post, user interactions are user2user
        self.train_post_inter, self.test_post_inter, \
        self.train_user_inter, self.test_user_inter = self.preprocessor.partition_data(self.week_num)

        logging.info('At week %d, recommend for upto week %d', self.week_num, self.eval_week)
        # users = [pid for pid in self.train_post_inter.keys()]  # Note that this gives all users, though some have 0 interactions
        self.users = set(self.train_post_inter.keys()).union(set(self.test_post_inter.keys()))
        logging.info('- Currently, %d users played in this forum', len(self.users))
        logging.info('- in total, %d users are enrolled in this forum', len(self.users))
        
        logging.info('Calculating the overall interactions...')
        self.all_post_inters = {pid: pd.concat([self.train_post_inter[pid], self.test_post_inter[pid]]) 
                                for pid in self.users}
        #print([len(x) for x in self.all_post_inters.values()])
        all_post_inters_df = pd.DataFrame()
        for pid, df in self.all_post_inters.items():
            df['PersonID'] = pid
            all_post_inters_df = all_post_inters_df.append(df)
        self.all_post_inters_df = all_post_inters_df
        #logging.debug(self.all_post_inters_df.Weeknum.unique())
        
        # Essentially a dictionary of dictionary, keys are the [pid]'s, 
        # because each [pid] correspond to a differnt test set
        for pid in self.users:
            self.eval_nids_per_person[pid] = self.extract_evaluation_inters_for_pid(pid) 
        
        all_notes = set()
        num_active_users = 0
        for pid, inter in self.train_post_inter.items():
            all_notes = all_notes.union(set(inter['NoteID'].unique()))
            if len(inter):
                num_active_users += 1
        logging.info('- the forum currently has %d posts', len(all_notes))
        logging.info('- %d users have made interactions', num_active_users)
        
        self.unrec_noteids = self.preprocessor.get_unsharable_posts()
        self.hierarchy = self.preprocessor.hierarchy
        self.all_note_contents = self.preprocessor.all_note_contents

    def extract_evaluation_inters_for_pid(self, pid):
        """
        Compute, for each user, the following two items:
        1. a complete log of interactions upto [week_num] + the log of interactions upto the week [week_num+stride] without the participation of [pid]
            the result is recorded in self.eval_inters_per_person, which is a dictionary of all users, each with a dataframe of interactions
        2. a complete list of all posts before [week_num] and all posts after [week_num] that are not created by [pid]
            the result is recorded in self.eval_nids_per_person, which is a dictionary of all users, each with a list of note ids
        """

        logging.debug('=======================================================================')
        # Get few additional weeks of interactions, ignoring all interactions created by [pid]
        logging.debug('Extracing interactions of all OTHER users WITHIN evaluation weeks for %d',pid)
        additional_week_inters = {user:inters.loc[inters.Weeknum.isin(self.eval_week)] 
                                  for user, inters in self.test_post_inter.items() if user != pid}
        #logging.info('Finished extraction, %d users has extracted, each has %s interactions', \
        #    len(additional_week_inters), str([len(x) for x in additional_week_inters.values()]))
        
        logging.debug('Appending to interactions of all users (including the active user) BEFORE evaluation weeks for %d',pid)
        eval_interactions = {}
        for user in self.users:
            if user in additional_week_inters.keys() and user in self.train_post_inter.keys():
                eval_interactions[user] = self.train_post_inter[user].append(additional_week_inters[user])
            elif user in additional_week_inters.keys():
                eval_interactions[user] = additional_week_inters[user]
                loggin.warning('user %d is new to the forum', user)
            elif user in self.train_post_inter.keys():
                eval_interactions[user] = self.train_post_inter[user]
                if user != pid: # [pid] wasn't there for sure.
                    logging.warning('user %d did not interact with the system in these weeks', user)
            else:
                logging.error('User# not consistent')

        self.eval_inters_per_person[pid] = eval_interactions
        #logging.debug(self.eval_inters_per_person[pid][pid].Weeknum.unique())
        logging.debug('Finished extracting evaluation set in interactions')

        visible_note_ids = set()
        # Get all available note_id's (including all notes created upto [week_num+1])
        for inter in eval_interactions.values():
            visible_note_ids = visible_note_ids.union(set(inter['NoteID'].values)) # keep adding up post ids

        logging.debug('User %d has in total %d posts in his evaluation set', pid, len(visible_note_ids))
        logging.debug('Finished extracting evaluation set in note ids')
        logging.debug('=======================================================================')
        
        return visible_note_ids
        
    def filter_out_evaluation_list_for_pid(self, pid):
        
        """
        Return a list of post node ids that can be recommended for [pid],
        Those posts meet the following criterion:
        1. They are not authored by the user [pid]
        2. They are not the direct replies to the user's [pid] original posts (Not Implemented Yet)
        
        Return a corresponding list of ground truth note ids

        """
           
        # Get all ground truth of [pid]'s interactions in test set
        logging.info("Extracting all ground truth of %d's interactions in test set ...", pid)
        gtp_inters_in_test = self.test_post_inter[pid] # from cutoff week to the end of course
        # Extract from it, the note_ids created by user in the additional weeks
        user_created_posts = set(gtp_inters_in_test.loc[(gtp_inters_in_test.weight == self.weight_post[0])
                                                       &(gtp_inters_in_test.Weeknum.isin(self.eval_week))].NoteID.unique())
        
        # Get the ground truth of [pid]'s interactions
        #  they are the interactions made by [pid] during the additional weeks
        gtp_inters_in_additional_week = None
        if self.gtp_criterion == 'explicit_only':
            gtp_inters_in_additional_week = gtp_inters_in_test.loc[(gtp_inters_in_test.weight<=self.cutoff_weight)&
                                                          (gtp_inters_in_test.Weeknum.isin(self.eval_week))]
        else:
            gtp_inters_in_additional_week = gtp_inters_in_test.loc[gtp_inters_in_test.Weeknum.isin(self.eval_week)]
        gtp = set(gtp_inters_in_additional_week.NoteID.unique()) # all interacted notes, including those already interacted

        logging.info("Extracting %d's already-interacted posts BEFORE the evaluation week...", pid)
        # To filter out posts that were already interacted by [pid]
        eval_list = set()
        # get a list of interactions by [pid] in training set, those already interacted posts are to be excluded from the evaluation set
        
        inter = self.train_post_inter[pid] 
        # all interacted posts (explicit+implicit)
        all_interacted_posts = set(inter['NoteID'].values)
        # Explicit interactions of the target user before eval week
        explicit_inters = inter.loc[inter['weight'] <= self.cutoff_weight]
        explicitly_interacted_posts = set(explicit_inters['NoteID'].values)
        implicitly_only_interacted_posts = all_interacted_posts - explicitly_interacted_posts
        # posts that were interacted recently (recently: current week or one week ago)
        recently_interacted_posts = set(inter.loc[inter['Weeknum']>=self.week_num-1]['NoteID'].values)
        # posts that were only implicitly interacted and not recently interacted
        not_recently_implicitly_only_interacted_posts = implicitly_only_interacted_posts - recently_interacted_posts # to be recommended

        # remove from all available posts the post that [pid] has interacted before
        eval_list = self.eval_nids_per_person[pid].difference(all_interacted_posts)\
                                                    .difference(set(self.unrec_noteids))\
                                                    .difference(user_created_posts)\
                                                    .union(not_recently_implicitly_only_interacted_posts)
        gtp = eval_list.intersection(gtp)
        
        if self.verbose:
            # This gives a set of note_ids of posts that are sharable and not interacted by [pid] before
            logging.debug('===========================================================================')
            logging.debug('Before week %d, user has made %d interactions to %d posts', self.week_num+1, len(inter), len(inter['NoteID'].unique()))
            train_week_inters = self.all_post_inters_df.loc[self.all_post_inters_df.Weeknum <= self.week_num]
            logging.debug('--- there were in total %d posts', len(train_week_inters['NoteID'].unique()))
            ### already_interacted = train_week_inters.loc[train_week_inters.PersonID==pid] ## 效果与上一句一样
            ### logging.debug('-- user made %d interactions to %d posts', len(already_interacted), len(already_interacted['NoteID'].unique()))
            logging.debug('From week %d to week %d:', self.week_num+1, self.week_num+1+self.week_stride)

            eval_week_inters = self.all_post_inters_df.loc[self.all_post_inters_df.Weeknum.isin(self.eval_week)]
            eval_week_new_posts = eval_week_inters.loc[eval_week_inters.weight==self.weight_post[0]]
            eval_week_new_posts_user_created = eval_week_new_posts.loc[eval_week_new_posts.PersonID==pid]
            logging.debug('-- %d new posts has been created, out of which %d are created by current user', \
                len(eval_week_new_posts), len(eval_week_new_posts_user_created)) 
            logging.debug('-- in total there were %d posts in the forum', len(self.all_post_inters_df.loc[self.all_post_inters_df.Weeknum<=self.week_num+1]['NoteID'].unique()))
            logging.debug('-- %d interactions has been made to %d posts', len(eval_week_inters), len(eval_week_inters['NoteID'].unique()))

            logging.debug('-- user made %d interactions to %d posts', \
                len(gtp_inters_in_additional_week), len(gtp_inters_in_additional_week['NoteID'].unique()))
            user_eval_week_inters = eval_week_inters.loc[eval_week_inters.PersonID==pid]
            ### logging.debug('user %d made %d interactions to %d posts', \ 
            ####     pid, len(user_eval_week_inters), len(user_eval_week_inters['NoteID'].unique())) # 与上一句效果一样
            already_interacted_posts = set(gtp_inters_in_additional_week['NoteID'].unique()).intersection(set(inter['NoteID'].unique()))
            logging.debug('---- out of which, %d was already interacted by user before week %d; %d are deleted/unsharable; %d were created by user', \
                len(already_interacted_posts), self.week_num+1, len(set(self.unrec_noteids).intersection()), len(user_created_posts))
            logging.debug('---- Excluding already-interacted posts, user created posts, and deleted....')
            logging.debug('---- Now, user has %d posts in the evaluation set, out of which %d are groud truths', len(eval_list), len(gtp))
            logging.debug('===========================================================================')

        return eval_list, not_recently_implicitly_only_interacted_posts, list(gtp)

    def run_popularity_eval(self, k):

        precisions = []     
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        for active_user in self.target_users:
            popularities = {nid:0 for nid in self.eval_nids_per_person[active_user]}
            for pid, inters in self.eval_inters_per_person[active_user].items(): 
                inter_counts = inters['NoteID'].value_counts().to_dict()
                popularities = {nid:count+inter_counts.get(nid,0) for nid, count in popularities.items()}
            
            eval_list, already_read_list, gtp = self.filter_out_evaluation_list_for_pid(active_user)
            eval_list = eval_list - already_read_list

            # print('%d / %d'%(len(eval_list), len(gtp)))

            rec_set = set(utils.get_popularity_rank(eval_list, popularities, k))

            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # print('--p@%d: %.3f'%(k, precision))
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0
        
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            # print('ap@%d: %.3f'%(k, ap))  
            aps.append(ap)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[active_user])
            # print(('miuf@%d: %.3f'%(k, miuf)))
            miufs.append(miuf)
            
            # self.true_prediction_labels(rec_set, gtp, active_user)

            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            # logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[active_user] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]

        return recs

    def run_random_eval(self, k):
        
        precisions = []
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        for active_user in self.target_users:

            user_specific_note_contents = self.all_note_contents.loc[self.all_note_contents['NoteID'].isin(self.eval_nids_per_person[active_user])] 
            eval_list, already_read_list, gtp = self.filter_out_evaluation_list_for_pid(active_user)
            # print('----------------------------')            
            eval_list = eval_list - already_read_list

            # print('%d / %d'%(len(eval_list), len(gtp)))
            # print('max precision@%d, %f'%(k, len(gtp)/k if len(gtp)/k<1 else 1.0))
            # print('----------------------------')
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0
             
            rec_set = set(utils.get_random_list(eval_list, k))

            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            aps.append(ap)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[active_user])
            miufs.append(miuf)
            

            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            diversities['semantic'].append(semd)
            logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[active_user] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]


        return recs


    def run_als_eval(self, k, mcf_settings):
        precisions = []    
        recalls = []    
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        mcf = MCF(mcf_settings)
        for pid in self.target_users:
            mcf.fit(self.eval_inters_per_person[pid])

            eval_list, read_list, gtp = self.filter_out_evaluation_list_for_pid(pid)
            eval_list = eval_list - read_list

            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            rec_set = mcf.run_user(pid, k)

            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)
            ap = utils.apk(list(gtp), list(rec_set), k)
            aps.append(ap)            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[pid])
            miufs.append(miuf)
            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            diversities['semantic'].append(semd)
            recs[pid] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]

        return recs



    def run_pure_ppr_eval(self, k, damp_factor):
        
        ppr = PurePPR(self.cutoff_weight, damp_factor) 
        
        
        precisions = []
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        for active_user in self.target_users:
            user_specific_note_contents = self.all_note_contents.loc[self.all_note_contents['NoteID'].isin(self.eval_nids_per_person[active_user])] 

        
            ppr.fit_user(active_user, self.users, self.eval_nids_per_person[active_user], 
                         self.eval_inters_per_person[active_user], user_specific_note_contents, 
                         self.week_num) 

            eval_list, already_read_list, gtp = self.filter_out_evaluation_list_for_pid(active_user)
            eval_list = eval_list - already_read_list

            ppr_values = ppr.run_user(active_user, eval_list)            
            note_ppr_values = {nid:value for nid, value in ppr_values.items() if nid in eval_list}
            rec_set = set([item[0] for item in utils.topn_from_dict(note_ppr_values, k)])
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # print('--p@%d: %.3f'%(k, precision))
            
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            # print('ap@%d: %.3f'%(k, ap))  
            aps.append(ap)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[active_user])
            # print(('miuf@%d: %.3f'%(k, miuf)))
            miufs.append(miuf)
            
            # self.true_prediction_labels(rec_set, gtp, active_user)
            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[active_user] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]

        return recs

    def run_coppr_eval(self, k, ppr_settings, pfilter, post_create_dates, temporal_start_week, ratio_read):
        """
        A pipeline to run CoPPR recommender offline each week
        """
        
        ''' 
            Init CoPPR '''
        ppr = CoPPR(self.cutoff_weight, ppr_settings) 
        # print('=== Step 1 Done ===')
        
        
        # User interactions
        # UserProfiler keeps analytical data for each user

        up = UserProfiler(self.train_user_inter, self.train_post_inter, self.users,
                          self.weight_user_lookup, self.weight_post_lookup, temporal_start_week)
        up.init_user_profiles(post_create_dates, self.week_num, slide_window_size = 2)

        
        precisions = []
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        for active_user in self.target_users:
            # print('================================================')
            # print('Predicting for user %d'%active_user)


            ''' 
                fit PPR engine with settings and training data
                 cotents of training posts, specifically for this user'''
            user_specific_note_contents = self.all_note_contents.loc[self.all_note_contents['NoteID'].isin(self.eval_nids_per_person[active_user])] 


            ppr.fit_user(active_user, self.users, self.eval_nids_per_person[active_user], 
                         self.eval_inters_per_person[active_user], user_specific_note_contents, 
                         self.week_num, up) 

            eval_list, already_read_list, gtp = self.filter_out_evaluation_list_for_pid(active_user)
            eval_list = eval_list - already_read_list

            # Build graph and run PPR for this user
            ppr_values = ppr.run_user(active_user, eval_list, self.precomputed_keywords)            
            # filtering and reranking the recommendations (Contextualized post-filtering)
            rec_set = pfilter.rerank(ppr_values, eval_list, already_read_list, None, k, ratio_read, verbose = 1) 

            # Evaluation
            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # print('--p@%d: %.3f'%(k, precision))
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            # print('ap@%d: %.3f'%(k, ap))  
            aps.append(ap)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[active_user])
            # print(('miuf@%d: %.3f'%(k, miuf)))
            miufs.append(miuf)
            
            # self.true_prediction_labels(rec_set, gtp, active_user)

            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            # logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[active_user] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]


        return recs, up

    def run_csclrec_eval(self, k, ppr_settings, pfilter, post_create_dates, temporal_start_week, ratio_read):
        """
        A pipeline to run CSCLRec recommender offline each week
        """
        
        '''  
            Init CSCLRec '''
        ppr = CSCLRec(self.cutoff_weight, ppr_settings) 
        # print('=== Step 1 Done ===')
        
        
        # User interactions
        # UserProfiler keeps analytical data for each user

        up = UserProfiler(self.train_user_inter, self.train_post_inter, self.users,
                          self.weight_user_lookup, self.weight_post_lookup, temporal_start_week)
        up.init_user_profiles(post_create_dates, self.week_num, slide_window_size = 2)
        ''' 
            Initialize the user profiler, and analyze each user's behavior for this time
            Precompute neareast neighbours for each user 
            (as a default setting to apply to the PPR graph for all users)'''
        # print('=== Step 2 Done ===')
        
        
        ''' 
        Init a content analyzer
        to compute the content graph consisting of wordnet hypernyms
        '''
        # Content analyzer
        #content_graph = None
        #if ppr_settings['cbf']:
        #content_analyzer = ContentAnalyzer(self.all_note_contents)
        #content_graph = content_analyzer.construct_hypernym_graph()
        content_graph = self.precomputed_content_graph
        # print('=== Step 3 Done ===')
        
        precisions = []
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        for active_user in self.target_users:
            # print('================================================')
            # print('Predicting for user %d'%active_user)


            ''' 
                fit CSCLRec engine with settings and training data
                 cotents of training posts, specifically for this user'''
            user_specific_note_contents = self.all_note_contents.loc[self.all_note_contents['NoteID'].isin(self.eval_nids_per_person[active_user])] 
            
            # Remove invisible posts from this user's evaluation graph
            user_specific_content_graph = content_graph.copy()
            removed_nodes = [x for x,y in content_graph.nodes(data=True) 
                             if y['node_type']=='post' and x not in self.eval_nids_per_person[active_user]]
            user_specific_content_graph.remove_nodes_from(removed_nodes)
            out_degree = user_specific_content_graph.degree()
            node_types = nx.get_node_attributes(user_specific_content_graph,'node_type')
            to_remove=[n for n in user_specific_content_graph 
                       if (out_degree[n] ==1 or out_degree[n] ==0) 
                       and node_types[n]=='hypernym']
            user_specific_content_graph.remove_nodes_from(to_remove)
            # print('----------------------------')
            # print('Content graph info')
            # print(nx.info(user_specific_content_graph))
        
            ppr.fit_user(active_user, self.users, self.eval_nids_per_person[active_user], 
                         self.eval_inters_per_person[active_user], user_specific_note_contents, 
                         self.week_num, up, user_specific_content_graph) 

            eval_list, already_read_list, gtp = self.filter_out_evaluation_list_for_pid(active_user)
            eval_list = eval_list - already_read_list
            # print('----------------------------')
            # print('%d / %d'%(len(eval_list), len(gtp)))
            # print('max precision@%d, %f'%(k, len(gtp)/k if len(gtp)/k<1 else 1.0))
            # print('----------------------------')

            # Build graph and run PPR for this user
            ppr_values = ppr.run_user(active_user, eval_list)            
            # filtering and reranking the recommendations (Contextualized post-filtering)
            rec_set = pfilter.rerank(ppr_values, eval_list, already_read_list, None, k, ratio_read, verbose = 1) 

            # Evaluation
            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # print('--p@%d: %.3f'%(k, precision))
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            # print('ap@%d: %.3f'%(k, ap))  
            aps.append(ap)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[active_user])
            # print(('miuf@%d: %.3f'%(k, miuf)))
            miufs.append(miuf)
            
            # self.true_prediction_labels(rec_set, gtp, active_user)

            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            # logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[active_user] = [rec_set, precision,recall, miuf, semd, sd, max_possible_recall]


        return recs, up

    def run_ucf_eval(self, k, cf_settings):
        precisions = []    
        recalls = []    
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}
        
        user_cf = UserCF(self.cutoff_weight, cf_settings)
        for pid in self.target_users:
            # logging.info('-----------------------------------------------')
            # logging.info('Prediction for user %d', pid)
            user_cf.fit(self.eval_inters_per_person[pid])

            eval_list, read_list, gtp = self.filter_out_evaluation_list_for_pid(pid)
            eval_list = eval_list - read_list

            # logging.info('%d / %d', len(eval_list), len(gtp))
            # logging.debug('max precision@%d, %f', k, len(gtp)/k if len(gtp)/k<1 else 1.0)
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            rec_set = user_cf.run_user(pid, eval_list, k)
            # logging.info('Recommendations: %s', str(rec_set))
            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # logging.info('p@%d: %.3f', k, precision)
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)
            ap = utils.apk(list(gtp), list(rec_set), k)
            aps.append(ap)
            # self.true_prediction_labels(rec_set, gtp, pid)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[pid])
            # print('miuf@%d: %.3f'%(k, miuf)) 
            miufs.append(miuf)

            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[pid] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]

        #print('map@%d'%k,np.mean(aps))

        return recs
    
    def run_cbf_eval(self, k, cbf_settings):
        """
        ['tfidf+lsi', 'tokens_phrases', 15, 'averaging','explicit_implicit','cosine']
        ['sentence_emb_precomputed', PRE_EMBS, PRE_SIMS, 'averaging','explicit_implicit','cosine']
        """
        precisions = []
        recalls = []
        aps = []
        miufs = []
        diversities = {'structural-1':[], 'structural-2':[], 'semantic':[]}
        recs = {}

        cbf = CBF(self.cutoff_weight, cbf_settings[0])

        for pid in self.target_users:
            logging.info('Prediction for user %d', pid)
            if cbf_settings[0]=='sentence_emb_precomputed':
                cbf.load_precomputed_emb(cbf_settings[1], self.eval_inters_per_person[pid])
            else:
                cbf.fit(self.eval_inters_per_person[pid], 5, ['NOUN', 'VERB'], STOP_WORDS, 
                        cbf_settings[1], cbf_settings[2], cbf_settings[6])
            cbf.construct_user_profile(pid, cbf_settings[3], cbf_settings[4])

            eval_list, read_list, gtp = self.filter_out_evaluation_list_for_pid(pid) 
            eval_list = eval_list - read_list
            max_possible_recall = k/len(gtp) if len(gtp)>k else 1.0

            #logging.info('%d / %d', len(eval_list), len(gtp))
            #logging.debug('max precision@%d, %f', k, len(gtp)/k if len(gtp)/k<1 else 1.0)
            # print('%d / %d'%( len(eval_list), len(gtp)))
            # print('max precision@%d, %f'%( k, len(gtp)/k if len(gtp)/k<1 else 1.0))
            
            # Feed the training set as usual to construct the testing RS
            rec_set = cbf.run_user(pid, eval_list, cbf_settings[5], k)

            precision = self.precision_at_k(rec_set, gtp)
            precisions.append(precision)
            # logging.info('p@%d: %.3f', k, precision)
            # print('p@%d: %.3f'%(k, precision))
            
            recall = self.recall_at_k(rec_set, gtp)
            recalls.append(recall)

            ap = utils.apk(list(gtp), list(rec_set), k)
            aps.append(ap)
            # self.true_prediction_labels(rec_set, gtp, pid)
            
            miuf = self.mean_inverse_user_frequency(rec_set, self.eval_inters_per_person[pid])
            # print('miuf@%d: %.3f'%(k, miuf)) 
            miufs.append(miuf)
            
            sd = self.structural_diversity(rec_set)
            semd = self.semantic_diversity(rec_set)
            # diversities['structural-1'].append(sd1)
            # diversities['structural-2'].append(sd2)
            diversities['semantic'].append(semd)
            # logging.info('Structural diversity@%d: %.3f',k, sd1)
            # logging.info('Structural diversity@%d: %.3f',k, sd2)
            # logging.info('Semantic diversity@%d: %.3f',k, semd)
            recs[pid] = [rec_set, precision, recall, miuf, semd, sd, max_possible_recall]

        return recs
    
    def set_ground_truth_criterion(self, option):
        self.gtp_criterion = option

    def precision_at_k(self, recs, gtp):
        if not len(gtp):
            return 1.0
        if len(recs):
            precision = len(recs.intersection(gtp))/len(recs)
        
            # print('Correct recommendations: %s' % str(recs.intersection(gtp)))
            return precision
        else:
            return 0.0
        
    def recall_at_k(self, recs, gtp):
        if len(gtp):
            recall = len(recs.intersection(gtp))/len(gtp)
        
            # print('Correct recommendations: %s' % str(recs.intersection(gtp)))
            return recall
        else:
            return 1.0

            
    def true_prediction_labels(self, recs, gtp, user):
        true_pred = recs.intersection(gtp)
        print(self.train_post_inter[user].columns)
        labels = self.test_post_inter[user].loc[(self.test_post_inter[user].NoteID.isin(true_pred)) 
                                            & (self.test_post_inter[user].Weeknum.isin(self.eval_week)), 
                                            ['NoteID','weight']] # To extract this user's interaction type in the week of evaluation
        former_labels = self.train_post_inter[user].loc[(self.train_post_inter[user].NoteID.isin(true_pred)) 
                                            & (self.train_post_inter[user].Weeknum.isin(self.eval_week)), 
                                            ['NoteID','weight','Weeknum']]
        for i, inter in labels.iterrows():
            #logging.info('--> Post %d is %s by the user', inter.loc['NoteID'], self.weight_post_lookup[inter.loc['weight']])
            print('- Post %d is %s by the user' % (inter.loc['NoteID'], self.weight_post_lookup[inter.loc['weight']]))
        for i, inter in former_labels.iterrows():
            print('- Post %d was %s by the user at week %d' % (inter.loc['NoteID'], 
                                                          self.weight_post_lookup[inter.loc['weight']], 
                                                          inter.loc['Weeknum']))
            
    def structural_diversity(self, recs):
        """
        Returns the number of direct replies and the number of sibling posts in a provided recommendation list
        """
        hierarchy = self.hierarchy
        direct_reply = []
        sibling_replies = []
        counts = 0
        for pair in itertools.combinations(recs, r=2):
                if not pair[0] == pair[1]:
                    direct_reply.append(utils.is_direct_reply(hierarchy, pair[0], pair[1]))
                    sibling_replies.append(utils.is_sibling(hierarchy, pair[0],pair[1]))
            #counts+=1
        if len(recs):
            return (sum(direct_reply)+sum(sibling_replies))
        else:
            return 0.0

    def semantic_diversity(self, recs):
        logging.info('[Evaluator] evaluating semantic diversity using model: universal-sentence-encoder-1')

        dist = []
        for pair in itertools.combinations(recs, r=2):
                if not pair[0] == pair[1]:
                    emb0 = self.precomputed_sen_embs[pair[0]]
                    emb1 = self.precomputed_sen_embs[pair[1]]
                    dist.append(utils.cosine_dist(emb0, emb1))
        return np.mean(dist)
    
    def mean_inverse_user_frequency(self, recs, all_interactions):
        """
        Item-based novelty measure
        Measures the inverse user frequency of each recommendation, aggregated on average
        """
        #self.eval_inters_per_person
        sum_iuf = 0
        for rec in recs:
            count = 0
            for user, inters in all_interactions.items():
                noteids = inters.NoteID.unique()
                if rec in noteids:
                    count+=1
            sum_iuf += np.log2(count/len(all_interactions))
        if len(recs):
            return -sum_iuf/len(recs)
        else:
            return 0
            
    def semantic_diversity_online(self, recs):
        rec_post_content = self.all_post_inters_df.loc[self.all_post_inters_df.NoteID.isin(recs)]['Contents'].to_list()

        logging.info('[Evaluator] evaluating semantic diversity using model: universal-sentence-encoder-1')
        sentence_embs = {}
        sentence_vectors = self.embed(rec_post_content)
        corr = np.inner(sentence_vectors, sentence_vectors) # inner product distance

        return np.mean(corr)