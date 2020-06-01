from UserProfiler import UserProfiler
import utils
import networkx as nx
import numpy as np
import pandas as pd
import logging
from collections import Counter

class CoPPR:
    def __init__(self, cutoff_weight, settings):
        self.users = None        
        self.all_items_per_user = None
        self.all_interactions_per_user = None
        self.sna_on = settings['sna']
        self.cbf_on = settings['cbf']
        self.hybrid_weights = settings.get('hybrid', [0.0,1.0,0.0,0.0])
        self.num_neighbours = settings.get('#neighbours', 5)
        self.temporal_weight_decay = settings.get('temporal_decay')
        self.damping_factor = settings.get('damp',0.85)

        self.curr_week = None
        self.cutoff_weight = cutoff_weight
        self.active_user = None 
        self.all_items_active_user = None 
        self.all_interactions_active_user = None
        self.all_note_contents_active_user = None 
        self.user_profiler = None

    def fit_user(self, active_user, users, all_items, all_interactions, all_contents, curr_week, up):
        """
        Fit CoPPR with training data
        """
        self.users = users
        self.active_user = active_user
        self.all_items_active_user = all_items
        self.all_interactions_active_user = all_interactions
        self.all_note_contents_active_user = all_contents
        self.curr_week = curr_week
        self.user_profiler = up

    def run_user(self, active_user, eval_list, precomputed_keywords):
        """
        Return a set of recommendations for the active user
        """
        #print(active_user)
        
        # Build graph
        self.precomputed_keywords = precomputed_keywords
        B, personalized_restart_nodes, item_nodes = self.build_graph(active_user)
        # Run PPR random walk
        ppr_values = nx.pagerank(G=B, 
                                 alpha=0.85, 
                                 personalization=personalized_restart_nodes)

        return ppr_values
        
    def build_graph(self, active_user):
        B = nx.DiGraph()
        
        # Add nodes
        B.add_nodes_from(self.all_items_active_user,node_type='post')         
        logging.debug('Graph for user %d has %d item nodes',active_user, len(self.all_items_active_user))
        B.add_nodes_from(self.users,node_type='user')        
        
        # Collect item nodes
        post_nodes = [] 
        for x,y in B.nodes(data=True):
            try:
                if y['node_type'] == 'post':
                    post_nodes.append(x)
            except KeyError:
                #print('Not a item node')
                pass
            
        total_edges = {'inters':[],"users":[],"items":[],'keywords':[]}
        
        # Add edges / weighted interactions
        personalized_restart_nodes = None
        temporal_decay = self.temporal_weight_decay
        for a_user in self.users:
            if self.sna_on:
                active_user_profile = self.user_profiler.get_user_profile(active_user)
                if active_user_profile.get('single_pass'):
                    temporal_decay = 0.85 # Lift temporal decay ratio to 0.85 for single pass users, which is essentially to decrease temporal decay to 0.15
            edge_set = self.get_user_to_post_edges(self.all_interactions_active_user[a_user], temporal_decay)            

            total_edges['inters'] += [(a_user, post,{'weight' : rel, 'type':'inters'}) 
                                      for post, rel in edge_set.items()]
            total_edges['inters'] += [(post, a_user,{'weight' : rel, 'type':'inters'}) 
                                      for post, rel in edge_set.items()]
            if a_user == active_user:
                personalized_restart_nodes = edge_set
                personalized_restart_nodes[active_user] = 1.0
        
        # Normalize edges
        total_edges['inters'] = self.normalize_edge_weights(total_edges['inters'], self.hybrid_weights[1])

        if self.sna_on: # if the learner profiler is turned on
            active_user_profile = self.user_profiler.get_user_profile(active_user)
            # nearest_user_neighbours --> edges
            if active_user_profile['new_user'] or active_user_profile['non_contributing']:
                # Connect to every one else in the community
                other_users = [usr for usr in self.users if usr != active_user]
                total_edges['users'] += [(active_user, user_id, {'weight' : 1/len(self.users), 'type':'users'}) 
                                             for user_id in other_users]
            elif active_user_profile.get('peripheral_tendency'):
                # add as default
                active_user_edges = self.user_profiler.find_knn(active_user, len(self.users)//self.num_neighbours)
                total_edges['users'] += [(active_user, ngb, {'weight' : sim, 'type':'users'}) 
                                              for (ngb, sim) in active_user_edges]                
            
                # add additional edges to recover lost readers
                total_edges['users'] += [(active_user, user_id, {'weight' : 1/len(self.users), 'type':'users'}) 
                                             for user_id in active_user_profile['lost_followers']]
            else:
                # Default setting
                active_user_edges = self.user_profiler.find_knn(active_user, len(self.users)//self.num_neighbours)
                total_edges['users'] += [(active_user, ngb, {'weight' : sim, 'type':'users'}) 
                                              for (ngb, sim) in active_user_edges]                
            total_edges['users'] = self.normalize_edge_weights(total_edges['users'], self.hybrid_weights[0])


        
        if self.cbf_on:
            # item_similarities --> edges
            
            total_edges['items'], total_edges['keywords'] = self.get_item_connections(B, [0.5, 0.5])
            #######print('# item edges:',len(total_edges['items']))
            total_edges['items'] = self.normalize_edge_weights(total_edges['items'], self.hybrid_weights[2])
            ########print('# keywords edges:',len(total_edges['keywords']))
            total_edges['keywords'] = self.normalize_edge_weights(total_edges['keywords'], self.hybrid_weights[3])
        

        for edges in total_edges.values():
            B.add_edges_from(edges)
        
        # finished graph building
        return B, personalized_restart_nodes, post_nodes
    
    
    
    def get_item_connections(self, graph, weight_decays):
        
        keywords_lookup = {note:kws for note, kws in self.precomputed_keywords.items() if note in self.all_items_active_user}

        vocab = set([word for words in keywords_lookup.values() for word in words])

        # Reference: https://datascience.stackexchange.com/a/40044
        co_occ = {ii:Counter({jj:0 for jj in vocab if jj!=ii}) for ii in vocab}
        k = 4

        for sen in keywords_lookup.values():
            #print(sen)
            for ii in range(len(sen)):
                if ii < k:
                    c = Counter(sen[0:ii+k+1])
                    del c[sen[ii]]
                    co_occ[sen[ii]] = co_occ[sen[ii]] + c
                elif ii > len(sen)-(k+1):
                    c = Counter(sen[ii-k::])
                    del c[sen[ii]]
                    co_occ[sen[ii]] = co_occ[sen[ii]] + c
                else:
                    c = Counter(sen[ii-k:ii+k+1])
                    del c[sen[ii]]
                    co_occ[sen[ii]] = co_occ[sen[ii]] + c

        # Having final matrix in dict form lets you convert it to different python data structures
        co_occ = {ii:dict(co_occ[ii]) for ii in vocab}

        keyword_nodes = list(co_occ.keys())
        graph.add_nodes_from(keyword_nodes, node_type='keyword')

        note2word_edges = [(nid, wrd, {'weight':1*weight_decays[0]}) 
                           for nid, wrds in keywords_lookup.items() for wrd in wrds]        
        note2word_edges += [(wrd, nid, {'weight':1*weight_decays[0]}) 
                           for nid, wrds in keywords_lookup.items() for wrd in wrds]
        word2word_edges = [(w1, w2, {'weight':weight*weight_decays[1]}) 
                           for w1, w2s in co_occ.items() for w2, weight in w2s.items()]
        word2word_edges += [(w2, w1, {'weight':weight*weight_decays[1]}) 
                           for w1, w2s in co_occ.items() for w2, weight in w2s.items()]
        
        return note2word_edges, word2word_edges
        
    def normalize_edge_weights(self, edges, edge_bias): 
        # Normalize edges, so that their mean is set to 1*edge_bias
        mean_edges = np.mean([edge[2]['weight'] for edge in edges])
        for edge in edges:
            edge[2]['weight'] = (edge[2]['weight']/mean_edges)*edge_bias
        
        return edges
    
    def check_validity(self, B, active_user, eval_list):
        logging.debug('Graph has %d edges', B.number_of_edges())
        for item in eval_list:
            edge = (active_user, item)
            if B.has_edge(*edge):
                logging.critical('**** Not valid %d', active_user)
                return
        logging.debug('Valid')

        
    def get_user_to_post_edges(self, df, temporal_decay):
        """
        Return the total weights of a certain user node to his items
        """
        df['edge_weight'] = np.where(df['weight'] < self.cutoff_weight, 
                                     temporal_decay**(self.curr_week - df['Weeknum']), 
                                     0.5*temporal_decay**(self.curr_week - df['Weeknum']))
        #df['edge_weight'] = df.apply(self.get_temporal_decay, axis=1)
        df = df[['NoteID','edge_weight']]
        return df.groupby('NoteID').sum().to_dict()['edge_weight']
        