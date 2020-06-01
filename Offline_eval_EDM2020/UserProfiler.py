import utils
import networkx as nx
import numpy as np
import pandas as pd
import operator



class UserProfiler(object):
    """
    User models that is designed to adapt learners by taking into consideration
    principles in the socio-collaborative learning context. The user profiler can 
    detect different types of learners and user behaviors in a knowledge 
    co-construction process within a educational forum. The output of this class
    can be used to design personalization strategies for these learners.
    """
    
    def __init__(self, user_interactions, post_interactions, users, weight_user_lookup, weight_post_lookup, TEMPORAL_START_WEEK):
        self.post_inter_lookup = {v:k for k,v in weight_post_lookup.items()} # weight look-up
        self.user_inter_lookup = {v:k for k,v in weight_user_lookup.items()} # weight look-up

        self.user_interactions = user_interactions
        self.post_interactions = post_interactions
        self.interactions_all = pd.concat(list(user_interactions.values()))
        self.interactions_all.PID_OUT = self.interactions_all.PID_OUT.astype('int64')
        self.users = users 


        self.explicit_inters = self.interactions_all.loc[self.interactions_all.weight < 4]
        self.explicit_uu_matrix, self.explicit_users = self.create_user_user_matrix(self.explicit_inters)
        
        self.user_interaction_graph = nx.MultiDiGraph()
        self.read_graph = nx.DiGraph()
        self.active_graph = nx.MultiDiGraph()
        self.TEMPORAL_START_WEEK = TEMPORAL_START_WEEK
    
        self.user_profiles = {pid:dict() for pid in self.users}
        
    
    def build_graph(self):
        
        edges = []
        read_edges = []

        for user in self.users:
            inters = self.user_interactions[user]
            
            # reply interactions
            reply_inters = inters.loc[inters.weight == self.user_inter_lookup['replied']]
            user_reply_edges = {}
            for r in zip(reply_inters['PID_IN'], reply_inters['Weeknum']):
                b_user = r[0]
                if b_user in user_reply_edges:
                    user_reply_edges[b_user]['counts']+=1
                    user_reply_edges[b_user]['contact_wks'].append(r[1])
                else:
                    user_reply_edges[b_user] = {'counts':1, 'contact_wks':[r[1]]}
            # Add to edges set
            for b_user, values in user_reply_edges.items():
                edge = (user, b_user, {'contact_wks' : values['contact_wks'], 
                                       'freq': values['counts'],
                                       'type':'reply', 
                                       'active-category':'active', 
                                       'knowledge-building': True})
                edges.append(edge)
                
            # link interactions
            link_inters = inters.loc[inters.weight == self.user_inter_lookup['linked']]
            user_link_edges = {}
            for r in zip(link_inters['PID_IN'], link_inters['Weeknum']):
                b_user = r[0]
                if b_user in user_link_edges:
                    user_link_edges[b_user]['counts']+=1
                    user_link_edges[b_user]['contact_wks'].append(r[1])
                else:
                    user_link_edges[b_user] = {'counts':1, 'contact_wks':[r[1]]}
            # Add to edges set
            for b_user, values in user_link_edges.items():
                edge = (user, b_user, {'contact_wks' : values['contact_wks'], 
                                       'freq': values['counts'],
                                       'type':'link', 
                                       'active-category':'active', 
                                       'knowledge-building': True})
                edges.append(edge)
                     
            # like interactions
            like_inters = inters.loc[inters.weight == self.user_inter_lookup['liked']]
            user_like_edges = {}
            for r in zip(like_inters['PID_IN'], like_inters['Weeknum']):
                b_user = r[0]
                if b_user in user_like_edges:
                    user_like_edges[b_user]['counts']+=1
                    user_like_edges[b_user]['contact_wks'].append(r[1])
                else:
                    user_like_edges[b_user] = {'counts':1, 'contact_wks':[r[1]]}
            # Add to edges set
            for b_user, values in user_like_edges.items():
                edge = (user, b_user, {'contact_wks' : values['contact_wks'], 
                                       'freq': values['counts'],
                                       'type':'like', 
                                       'active-category':'active', 
                                       'knowledge-building': False})
                edges.append(edge)
                
            # read/anonymous read interactions
            read_inters = inters.loc[(inters.weight == self.user_inter_lookup['read']) 
                                      | (inters.weight == self.user_inter_lookup['anonymously read'])]
            user_read_edges = {}
            for r in zip(read_inters['PID_IN'], read_inters['Weeknum']):
                b_user = r[0]
                if b_user in user_read_edges:
                    user_read_edges[b_user]['counts']+=1
                    user_read_edges[b_user]['contact_wks'].append(r[1])
                else:
                    user_read_edges[b_user] = {'counts':1, 'contact_wks':[r[1]]}
            # Add to edges set
            for b_user, values in user_read_edges.items():
                edge = (user, b_user, {'contact_wks' : values['contact_wks'], 
                                       'freq': values['counts'],
                                       'type':'read', 
                                       'active-category':'passive', 
                                       'knowledge-building': False})
                edges.append(edge)
                read_edges.append(edge)
        
        # Add nodes and edges
        self.user_interaction_graph.add_nodes_from(self.users)     
        self.user_interaction_graph.add_edges_from(edges)
        self.read_graph.add_nodes_from(self.users)     
        self.read_graph.add_edges_from(read_edges)
    
    def get_special_user_profiles(self):
        new_users = []
        peripheral_users = []
        single_pass_users = []
        non_contributing_users = []
        for user_id, up in self.user_profiles.items():
            if up.get('new_user'):
                new_users.append(user_id)
            if up.get('peripheral_tendency'):
                peripheral_users.append(user_id)
            if up.get('single_pass'):
                single_pass_users.append(user_id)
            if up.get('non_contributing'):
                non_contributing_users.append(user_id)

        return {'new_users':new_users,'peripheral_users':peripheral_users,'single_pass_users':single_pass_users,'non_contributing_users':non_contributing_users}

    def get_user_profile(self, user_id):
        return self.user_profiles[user_id]
    
    def init_user_profiles(self, post_creation_date_lookup, curr_wk, slide_window_size = 2):
        self.build_graph()

        outdegrees = self.user_interaction_graph.out_degree() # a MultiDiGraph
        read_degrees = self.read_graph.out_degree()
        
        active_graph = nx.MultiDiGraph([(u,v,e) for u,v,e in self.user_interaction_graph.edges(data=True) 
                        if e['active-category']=='active']) ## select only explicit actions (like, reply, link)
        self.active_graph = active_graph
        active_out_degrees = active_graph.out_degree()
        for user in self.users:
            if user not in active_graph:
                active_graph.add_node(user)
        
        '''
        Deal with temporal dynamics with a sliding window
            Separate recent interactions and dated interactions 
        ''' 
        temporal_subgraph_edges = {'recent_active':[],'recent_inactive':[],
                                   'old_active':[],'old_inactive':[]}
        for u,v,e in self.user_interaction_graph.edges(data=True):
            if max(e['contact_wks']) > curr_wk-slide_window_size: # recent wk: curr_wk, ..., curr_wk-wind_size+1
                if e['active-category']=='active':
                    ## Keep only active/explicit interactions during recent weeks
                    temporal_subgraph_edges['recent_active'].append((u,v,e))
            elif [wk 
                  for wk in e['contact_wks'] 
                  if wk in range(curr_wk-slide_window_size, curr_wk-2*slide_window_size, -1)]:
                ## Keep only active/explicit interactions that was occured in the last time window
                if e['active-category']=='active':
                    temporal_subgraph_edges['old_active'].append((u,v,e))

        recent_active_graph = nx.MultiDiGraph(temporal_subgraph_edges['recent_active']) # a MultiDiGraph
        old_active_graph = nx.MultiDiGraph(temporal_subgraph_edges['old_active']) # a MultiDiGraph
        for user in self.users:
            if user not in recent_active_graph:
                recent_active_graph.add_node(user) # add isolated users if no edges were included in this subgraph
            if user not in old_active_graph:
                old_active_graph.add_node(user)
        
        for user in self.users:
            # print('user',user)
            ''' New user '''
            if outdegrees[user] <= 0:
                # New user
                # -> Completely no interactions
                self.user_profiles[user]['new_user'] = True
            else:
                self.user_profiles[user]['new_user'] = False
            
            ''' Cold start user ''' # Not used
            if active_out_degrees[user] < 5:
                # Cold start user
                self.user_profiles[user]['cold_start'] = True
            else:
                self.user_profiles[user]['cold_start'] = False
                
            post_inters = self.post_interactions[user]
            # Post creation behavior
            user_created_posts = post_inters.loc[post_inters.weight == self.post_inter_lookup['created'], 'Weeknum']

            # Post re-visit behavior
            user_revisit_posts = post_inters.loc[post_inters.weight == self.post_inter_lookup['revisited'], 'Weeknum']
            
            self.user_profiles[user]['#creations'] = len(user_created_posts)
            self.user_profiles[user]['#revisits'] = len(user_revisit_posts)
            
            ''' Non-contributing user '''            
            if self.user_profiles[user]['#creations']==0:
                # not contributing users
                # -> no creation, no replies
                if self.user_profiles[user]['#revisits'] > 5:
                    self.user_profiles[user]['non_contributing'] = 'revisiting'
                else:
                    self.user_profiles[user]['non_contributing'] = 'non_revisiting'
            else:
                self.user_profiles[user]['non_contributing'] = False

            if curr_wk >= self.TEMPORAL_START_WEEK:
                ''' Single pass behavior'''
                # Check if user has read 
                #     posts created before the last week (1, curr_wk-1) 
                # during time from last week to now (curr_wk-1, curr_wk)
                user_read_posts = post_inters.loc[(post_inters.weight == self.post_inter_lookup['read']) 
                                                  & (post_inters.Weeknum.isin([curr_wk, curr_wk-1])), 'NoteID'].values
                read_old_posts = [post_creation_date_lookup.get(nid, curr_wk)<curr_wk-1 for nid in user_read_posts]
                # print('posts that this user recently read:', user_read_posts)
                # print('whether these posts were created long ago', read_old_posts)
                if True in read_old_posts:
                    self.user_profiles[user]['single_pass'] = False
                else:
                    self.user_profiles[user]['single_pass'] = True

                '''Peripheral tendency'''
                # Check if user has lost 
                #     more than half their readers before the last week (1, curr_wk-1) 
                # during the time period from last week to now (curr_wk-1, curr_wk)
                curr_followers = set(recent_active_graph.predecessors(user))                
                prev_followers = set(old_active_graph.predecessors(user))
                if len(curr_followers) < 0.5 * len(prev_followers):
                    self.user_profiles[user]['peripheral_tendency'] = True
                    self.user_profiles[user]['lost_followers'] = prev_followers - curr_followers
                    self.user_profiles[user]['new_followers'] = curr_followers - prev_followers
                else:
                    self.user_profiles[user]['peripheral_tendency'] = False
                    self.user_profiles[user]['lost_followers'] = prev_followers - curr_followers
                    self.user_profiles[user]['new_followers'] = curr_followers - prev_followers
                            
    def create_user_user_matrix(self, interactions):
        matrix_dict = {}
        users = set(interactions.PID_OUT.unique()).union(set(interactions.PID_IN.unique()))
        for user1 in users:
            matrix_dict[user1] = {}
            for user2 in users:
                matrix_dict[user1][user2] = 0
        for i, row in interactions.iterrows():
            p1 = row.PID_OUT
            p2 = row.PID_IN
            matrix_dict[p1][p2] += 1
        return pd.DataFrame.from_dict(matrix_dict), users

    def find_knn(self, pid, k):
        """
        return the [k] most familiar users to user [pid]
        """
        uu_matrix = self.explicit_uu_matrix
        users = self.explicit_users
            
        distance = {}
        other_users = [usr for usr in users if usr!=pid]
        try:
            p1 = uu_matrix[pid].to_dict()
            sum_degree = np.sum(list(p1.values()))
            degree_connect = sorted(p1.items(), key=operator.itemgetter(1), reverse=True)
            degree_connect = [(usr, deg/sum_degree) for usr, deg in degree_connect if deg>0]
            #print('deg con', degree_connect)
            try:
                return degree_connect[:k]
            except IndexError:
                print('User', pid,'does not have enough history. Let us help him/her to find something interesting')
                return [(usr, 1) for usr in other_users]
        except KeyError as e:
            print('User', pid,'does not have any history. Let us help him/her to find something interesting')
            return [(usr, 1) for usr in other_users]