import utils
import networkx as nx
import numpy as np
import pandas as pd

class PurePPR:
    def __init__(self, cutoff_weight, damp_factor):
        self.users = None        
        self.all_items_per_user = None
        self.all_interactions_per_user = None
        
        self.curr_week = None
        #print(self.hybrid_weights)
        self.damp_factor = damp_factor
        self.cutoff_weight = cutoff_weight
        self.active_user = None
        self.all_items_active_user = None 
        self.all_interactions_active_user = None 
        self.all_note_contents_active_user = None 
        

    def fit_user(self, active_user, 
                 users, all_items, all_interactions, all_contents, 
                 curr_week):

        self.users = users
        self.active_user = active_user
        self.all_items_active_user = all_items
        self.all_interactions_active_user = all_interactions
        self.all_note_contents_active_user = all_contents
        self.curr_week = curr_week
        
    def run_user(self, active_user, eval_list):
        """
        Return a set of recommendations for the active user
        """

        B, personalized_restart_nodes, item_nodes = self.build_graph(active_user)
        # Run PPR
        ppr_values = nx.pagerank(G=B, 
                                 alpha=self.damp_factor, 
                                 personalization=personalized_restart_nodes)

        return ppr_values
        
    def build_graph(self, active_user):
        B = nx.DiGraph()
        
        # Add nodes
        B.add_nodes_from(self.all_items_active_user,node_type='post')         
        #logging.debug('Graph for user %d has %d item nodes',active_user, len(self.all_items_active_user))
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
            
        total_edges = {'user2post':[]}

        
        # Add edges / weighted interactions
        personalized_restart_nodes = None
        for a_user in self.users:
            edge_set = self.get_user_to_post_edges(self.all_interactions_active_user[a_user])            

            total_edges['user2post'] += [(a_user, post,{'weight' : rel, 'type':'user2post'}) 
                                      for post, rel in edge_set.items()]
            total_edges['user2post'] += [(post, a_user,{'weight' : rel, 'type':'user2post'}) 
                                      for post, rel in edge_set.items()]
            if a_user == active_user:
                personalized_restart_nodes = edge_set
                personalized_restart_nodes[active_user] = 1.0
        
        for edges in total_edges.values():
            B.add_edges_from(edges)
        
        
        # print('Total #edges (all):', output.number_of_edges())
        # finished graph building
        return B, personalized_restart_nodes, post_nodes
    

    def get_user_to_post_edges(self, df):
        df['edge_weight'] = np.where(df['weight'] < self.cutoff_weight, 1, 0.5)
        df = df[['NoteID','edge_weight']]
        return df.groupby('NoteID').sum().to_dict()['edge_weight']
        
    
