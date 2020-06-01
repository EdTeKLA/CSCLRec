import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pickle
import csv
import implicit
import itertools
import copy
import os
import networkx as nx
import operator

import sys
import os
import random
from collections import Counter
import itertools
import utils
from sklearn.preprocessing import MinMaxScaler

"""
References:
https://github.com/benfred/implicit
https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
https://jessesw.com/Rec-System/
"""

class MCF():

    def __init__(self, wmf_settings):
        self.inter_df = None
        self.alpha = wmf_settings[0]
        self.mid_to_idx = {}
        self.idx_to_mid = {}
        self.uid_to_idx = {}
        self.idx_to_uid = {}
        
    def run_user(self, pid, k):
        """
        Return a set of recommendations for the user
        """
        user_vecs, item_vecs = implicit.alternating_least_squares((self.inter_df*self.alpha).astype('double'), 
                                                                  factors=64, 
                                                                  regularization = 0.1, 
                                                                  iterations = 50)
        rec_list = self.rec_items(pid, self.inter_df, user_vecs, item_vecs, num_items = k)
        return set(rec_list)
    

    def fit(self, post_interactions):
        self.post_interactions = post_interactions
        lst_tmp = []
        for v, k in self.post_interactions.items():
            df_tmp = k.copy()
            df_tmp['PersonID'] = v
            lst_tmp.append(df_tmp)
        inter_df = pd.concat(lst_tmp)
        inter_df = inter_df.drop(columns=['Contents','AwlCount','Weeknum','weight'])
        self.inter_df = self.construct_sparse_matrix(inter_df)
            
    
    def construct_sparse_matrix(self, df):
        n_users = df.PersonID.unique().shape[0]
        n_items = df.NoteID.unique().shape[0]

        # Create mappings
        self.mid_to_idx = {}
        self.idx_to_mid = {}
        for (idx, mid) in enumerate(df.NoteID.unique().tolist()):
            self.mid_to_idx[mid] = idx
            self.idx_to_mid[idx] = mid

        self.uid_to_idx = {}
        self.idx_to_uid = {}
        for (idx, uid) in enumerate(df.PersonID.unique().tolist()):
            self.uid_to_idx[uid] = idx
            self.idx_to_uid[idx] = uid
                
        I = df.PersonID.apply(self.map_ids, args=[self.uid_to_idx]).as_matrix()
        J = df.NoteID.apply(self.map_ids, args=[self.mid_to_idx]).as_matrix()
        V = np.ones(I.shape[0])
        M = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
        M = M.tocsr()
        
        return M
            
    def map_ids(self, row, mapper):
        return mapper[row]
    
    def rec_items(self, pid, mf_train, user_vecs, item_vecs, num_items = 10):
        """
        Credit to https://jessesw.com/Rec-System/
        """
        try:
            cust_ind = self.uid_to_idx[pid] # Returns the index row of our customer id
            pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
            pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
            pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
            rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot product of user vector and all item vectors
            # Scale this recommendation vector between 0 and 1
            min_max = MinMaxScaler()
            rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
            recommend_vector = pref_vec*rec_vector_scaled 
            # Items already purchased have their recommendation multiplied by zero
            product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
            # of best recommendations
            rec_list = [] # start empty list to store items
            for index in product_idx:
                code = self.idx_to_mid[index]
                rec_list.append(code) 
                # Append our descriptions to the list
            return rec_list
        except KeyError:
            print("User",pid,"does not have interactions, thus no predictions can be done")
            return []