from ContentAnalyzer import ContentAnalyzer
from UserProfiler import UserProfiler
import utils
import networkx as nx
import numpy as np
import pandas as pd
import logging

class CSCLRec:
    def __init__(self, cutoff_weight, settings):
        self.users = None        
        self.all_items_per_user = None
        self.all_interactions_per_user = None
        self.sna_on = settings['sna']
        self.cbf_on = settings['cbf']
        self.hybrid_weights = settings.get('hybrid', [0.0,1.0,0.0])
        self.num_neighbours = settings.get('#neighbours', 5)
        self.temporal_weight_decay = settings.get('temporal_decay', 0.85)
        self.damping_factor = settings.get('damp',0.85)
        
        self.curr_week = None
        #print(self.hybrid_weights)
        self.cutoff_weight = cutoff_weight
        self.active_user = None
        self.all_items_active_user = None
        self.all_interactions_active_user = None
        self.all_note_contents_active_user = None 
        
        self.user_profiler = None
        self.content_graph = None

    def fit_user(self, active_user, 
                 users, all_items, all_interactions, all_contents, 
                 curr_week, user_profiler, content_graph):
        """
        Fit the CSCLRec with training data
        """
        self.users = users
        self.active_user = active_user
        self.all_items_active_user = all_items
        self.all_interactions_active_user = all_interactions
        self.all_note_contents_active_user = all_contents
        self.curr_week = curr_week
        self.user_profiler = user_profiler
        self.content_graph = content_graph
        
    def run_user(self, active_user, eval_list):
        """
        Return a set of recommendations for the active user
        """
        #print(active_user)
        
        # Build graph
        B, personalized_restart_nodes, item_nodes = self.build_graph(active_user)
        # Run PPR
        ppr_values = nx.pagerank(G=B, 
                                 alpha=self.damping_factor, 
                                 personalization=personalized_restart_nodes)
        
        #print(rec_set)
        
        # Validate result
        #self.check_validity(B, active_user, eval_list)

        return ppr_values
        
    def build_graph(self, active_user):
        B = nx.DiGraph()
        
        # Add nodes
        B.add_nodes_from(self.all_items_active_user,node_type='post')         
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
            
        total_edges = {'user2post':[],"user2user":[],"items":[]}

        if self.sna_on:
            active_user_profile = self.user_profiler.get_user_profile(active_user)
            # nearest_user_neighbours --> edges
            if active_user_profile['new_user'] or active_user_profile['non_contributing']:
                # Connect to every one else in the community
                other_users = [usr for usr in self.users if usr != active_user]
                total_edges['user2user'] += [(active_user, user_id, {'weight' : 1/len(self.users), 'type':'user2user'}) 
                                             for user_id in other_users]
            elif active_user_profile.get('peripheral_tendency'):
                # add as default
                active_user_edges = self.user_profiler.find_knn(active_user, len(self.users)//self.num_neighbours)
                total_edges['user2user'] += [(active_user, ngb, {'weight' : sim, 'type':'user2user'}) 
                                              for (ngb, sim) in active_user_edges]                
            
                # add additional edges to recover lost readers
                total_edges['user2user'] += [(active_user, user_id, {'weight' : 1/len(self.users), 'type':'user2user'}) 
                                             for user_id in active_user_profile['lost_followers']]
            else:
                # Default setting
                active_user_edges = self.user_profiler.find_knn(active_user, len(self.users)//self.num_neighbours)
                total_edges['user2user'] += [(active_user, ngb, {'weight' : sim, 'type':'user2user'}) 
                                              for (ngb, sim) in active_user_edges]                

            total_edges['user2user'] = self.normalize_edge_weights(total_edges['user2user'], self.hybrid_weights[0])

        
        # Add edges / weighted interactions
        personalized_restart_nodes = None
        temporal_decay = self.temporal_weight_decay
        for a_user in self.users:
            if self.sna_on:
                active_user_profile = self.user_profiler.get_user_profile(active_user)
                if active_user_profile.get('single_pass'):
                    temporal_decay = 0.85 # Lift temporal decay ratio to 0.85 for single pass users, which is essentially to decrease temporal decay to 0.15
            edge_set = self.get_user_to_post_edges(self.all_interactions_active_user[a_user], temporal_decay)            

            total_edges['user2post'] += [(a_user, post,{'weight' : rel, 'type':'user2post'}) 
                                      for post, rel in edge_set.items()]
            total_edges['user2post'] += [(post, a_user,{'weight' : rel, 'type':'user2post'}) 
                                      for post, rel in edge_set.items()]
            if a_user == active_user: # Set restart nodes as the active user node and the post nodes connecting to them
                personalized_restart_nodes = edge_set
                personalized_restart_nodes[active_user] = 1.0

        
        # Normalize edges
        total_edges['user2post'] = self.normalize_edge_weights(total_edges['user2post'], self.hybrid_weights[1])

        for edges in total_edges.values():
            B.add_edges_from(edges)
        
        if self.cbf_on:

            for u,v,d in self.content_graph.edges(data=True):
                d['weight'] *= self.hybrid_weights[2]
            output = nx.compose(B, self.content_graph)
        else:
            output = B
        
        return output, personalized_restart_nodes, post_nodes
    
    
    def normalize_edge_weights(self, edges, edge_bias): 
        mean_edges = np.mean([edge[2]['weight'] for edge in edges])
        for edge in edges:
            edge[2]['weight'] = (edge[2]['weight']/mean_edges)*edge_bias
        
        edges_set = [edge[2]['weight'] for edge in edges]
        
        return edges
    
    def check_validity(self, B, active_user, eval_list):
        logging.debug('Graph has %d edges', B.number_of_edges())
        for item in eval_list:
            edge = (active_user, item)
            if B.has_edge(*edge):
                logging.critical('**** Not valid %d', active_user)
                print('**** Not valid %d', active_user)
                return
        logging.debug('Valid')

    def get_user_to_post_edges(self, df, temporal_decay):
        """
        Return the total weights of a certain user node to his items
        """
        df['edge_weight'] = np.where(df['weight'] < self.cutoff_weight, 
                                     temporal_decay**(self.curr_week - df['Weeknum']), 
                                     0.5*temporal_decay**(self.curr_week - df['Weeknum']))
        df = df[['NoteID','edge_weight']]
        return df.groupby('NoteID').sum().to_dict()['edge_weight']
        
    
