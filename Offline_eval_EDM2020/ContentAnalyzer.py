import numpy as np
import pandas as pd
import networkx as nx
from STOPWORDS import STOP_WORDS as CUSTOM_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
import utils
import spacy
import re
from nltk.corpus import wordnet as wn   
from nltk.wsd import lesk

class ContentAnalyzer():
    """
    A module of CSCLRec and CoPPR
    Analyzing forum posts' semantics utilizing the WordNet and its hypernyms.
    """
    def __init__(self, all_note_contents):
        self.all_note_contents = all_note_contents
        self.all_note_dict = all_note_contents.set_index('NoteID').to_dict()['Contents']
        self.note_ids = []
        self.data_words = {}
        self.post2hyps = {}
        
    def preprocess(self, tags):
        nlp = spacy.load('en_core_web_md')
        for nid, content in self.all_note_dict.items():
            url_regx = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
            email_regx = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
            at_regx = r"@.*"
            
            content = re.sub(url_regx, '', content) # replace url
            content = re.sub(email_regx, '', content) # replace email
            content = re.sub(at_regx, '', content) # replace @ statement with empty string

            doc = nlp(content)
            stop_list = set(list(CUSTOM_STOP_WORDS)+list(STOP_WORDS))
            sent_out = [tk.lemma_ for tk in doc 
                        if tk.text.strip() 
                         and tk.lemma_ not in stop_list # remove from customized stop list
                         and not tk.is_punct # remove punctuations
                         and not tk.like_email # remove emails
                         and not tk.like_url # remove url 
                         and tk.pos_ in tags
                         and tk.ent_type_ not in ['PERSON','PERCENT','MONEY','QUANTITY', 
                                                  'ORDINAL','CARDINAL','DATE','TIME']]
            self.data_words[nid] = sent_out 
        self.note_ids = list(self.data_words.keys())
        
    def extract_keywords(self, keyword_nums):
        tfidf_matrix, tf_dicts, post_appear_dict = utils.tfidf(self.data_words.values())
        if keyword_nums>1:
            keywords = {i: utils.get_top_tfidfs(tfidf_matrix[i], keyword_nums)
                        for i, m in enumerate(tfidf_matrix)}
        else:
            keyword_ratio = 1/keyword_nums
            keywords = {i: utils.get_top_tfidfs(tfidf_matrix[i], len(tfidf_matrix[i])//keyword_ratio) 
                        for i, m in enumerate(tfidf_matrix)}
        self.keywords = {self.note_ids[i]:kws for i,kws in keywords.items()}
    
    def retrieve_hypernyms(self, keyword, sent, option='lesk'):
        # as all keywords are nouns
        if option=='exhaustive':
            kw_synsets = wn.synsets(keyword, pos=wn.NOUN)
            kw_hypernyms = []
            for kw in kw_synsets:
                kw_hypernyms += kw.hypernyms()
        elif option=='lesk':
            lemma = lesk(sent, keyword, 'n')
            if lemma:
                kw_hypernyms = lemma.hypernyms()
                if kw_hypernyms:
                    return kw_hypernyms[-1]
                else:
                    return lemma
            else:
                return lemma
    
    def construct_hypernym_graph(self):
        '''
        To generate the post-to-hypernym graph, as a subgraph of the PPR graph
        '''
        edges = []
        nid_nodes = []
        hypernyms = {} 
        post2hyps = {}
        for nid, kws in self.keywords.items():
            post_hypernyms = [] # a list to keep hypernyms correspond to this post
            for kw in kws:
                kw_hypernym = self.retrieve_hypernyms(kw, self.all_note_dict[nid], option='lesk')
                if kw_hypernym:
                    post_hypernyms.append(kw_hypernym)
                if kw_hypernym in hypernyms:
                    hypernyms[kw_hypernym] +=1
                else:
                    hypernyms[kw_hypernym] = 1
            nid_nodes.append(nid)
            post2hyps[nid] = set(post_hypernyms) # remove duplicates
            edges += [(nid, hyp) for hyp in post_hypernyms]
        
        G = nx.Graph()
        G.add_nodes_from(nid_nodes, node_type='post')
        G.add_nodes_from(list(hypernyms.keys()), node_type='hypernym')
        G.add_edges_from(edges, edge_type='post2hypernym',weight=1)
        # Take a look at the isolated nodes
        # print('isolates before removing', list(nx.isolates(G)))
        
        out_degree = G.degree()
        node_types = nx.get_node_attributes(G,'node_type')
        to_remove=[n for n in G if out_degree[n] ==1 and node_types[n]=='hypernym']
        to_remove.append(None)
        G.remove_nodes_from(to_remove)
        # Take a look at the isolates, should now only contain post nodes, those posts without hypernyms
        #print('isolates after removing', list(nx.isolates(G)))
        self.post2hyps = post2hyps
        
        return G.to_directed()    

        
    def display_hyps(self, sav = False):
        if sav:
            with open('hyps.txt', 'w') as f:
                for nid, hyps in self.post2hyps.items():
                    print(nid, file=f)
                    print(self.all_note_dict[nid], file=f)
                    print(hyps, file=f)
                    print(file=f)
        else:
            for nid, hyps in self.post2hyps.items():
                print(nid)
                print(self.all_note_dict[nid])
                print(hyps)
                print()
