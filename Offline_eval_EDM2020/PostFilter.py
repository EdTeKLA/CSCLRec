import numpy as np
import pandas as pd
from STOPWORDS import STOP_WORDS as CUSTOM_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
import utils
import spacy
import re
import math

class PostFilter:
    def __init__(self):
        self.data = {}
        self.data_raw = {}
        self.df = None
        self.awls = {}
        self.unrec_noteids = None

    def fit(self, data):
        self.df = data
        nlp = spacy.load('en_core_web_md')
        for i, row in data.iterrows():
            nid = row['NoteID']
            content = row['Contents']
            awl = row['AwlCount']
            self.data_raw[nid] = content
            url_regx = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
            email_regx = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
            at_regx = r"@.*"
            
            content = re.sub(url_regx, ' URL.', content) # replace url
            content = re.sub(email_regx, ' EMAIL.', content) # replace email
            content = re.sub(at_regx, '', content) # replace @ statement with empty string

            doc = nlp(content)
            stop_list = set(list(CUSTOM_STOP_WORDS)+list(STOP_WORDS))
            sent_out = [tk.text for tk in doc 
                        if tk.text.strip() 
                         and tk.lemma_ not in stop_list # remove from customized stop list
                         and not tk.is_punct # remove punctuations
                         and not tk.like_email # remove emails
                         and not tk.like_url # remove url 
                         and tk.pos_ in ['NOUN','PROPN']
                         and tk.ent_type_ not in ['PERSON','PERCENT','MONEY','QUANTITY', 
                                                  'ORDINAL','CARDINAL','DATE','TIME']]
            self.data[nid] = set(sent_out)
            self.awls[nid] = awl
            
        self.unrec_noteids = set()

    def display_prune_results(self, sav = False):
        if sav:
            with open('out.txt', 'w') as f:
                for nid, content_bow in self.data.items():

                    print(nid, file=f)
                    print(self.data_raw[nid], file=f)
                    print(content_bow, file=f)
                    if nid in self.unrec_noteids:
                        print('--filtered--', file=f)
                    print(file=f)
        else:
            for nid, content_bow in self.data.items():
                print(nid)
                print(self.data_raw[nid])
                print(content_bow)
                if nid in self.unrec_noteids:
                    print('--filtered--')
                print()
            
    def get_unsharables(self):
        return self.unrec_noteids

    def prune_on_length(self, thres):
        for nid, content_bow in self.data.items():
            if len(content_bow)<thres:
                self.unrec_noteids.add(nid)

    def prune_on_awl(self, thres):
        for nid, count in self.awls.items():
            if count<thres:
                self.unrec_noteids.add(nid)
                
    def rerank(self, ppr_values, eval_list, already_read_list, up, k, ratio, verbose = 0):
        # filter and rank ppr values of all items in the eval list
        rec_set = self.get_ppr_recommendations(ppr_values, eval_list, already_read_list, k=k, ratio_read=ratio)        
        # Generate a set of post ids
        recs = set([node_id for (node_id,value) in rec_set])

        output = []
        num_rec = 0
        for rec in recs:
            if rec not in self.get_unsharables():
                output.append(rec)
                num_rec+=1
        
        try:
            return set(output[:k])
        except IndexError as e:
            print('Less than %d recommendations are generated'%k)
            return set(output)

    
    def prune_on_keywords(self, thres):
        data_words, note_ids, id2word, corpus = utils.preprocess(self.df, 5, ['NOUN','VERB'], STOP_WORDS, 'tokens_phrases')
        tfidf_matrix, tf_dicts, post_appear_dict = utils.tfidf(data_words)

        keywords = {i: utils.get_tfidfs_thres(tfidf_matrix[i], thres)
                        for i, m in enumerate(tfidf_matrix)}
        word2id = {v: k for k, v in id2word.items()}
        tfidf_corpus = [[(word2id[pair[0]], pair[1]) for pair in post.items()] for post in keywords]


    def prune_direct_replies(self, hierarchy, user_created_posts):
        """
        A helper function to remove from the uninteracted notes those posts that are the
        direct replies of the user's created posts
        """
        direct_replies = set()
        for note in user_created_posts:
            if note in hierarchy:  # in case that the post is not archived yet
                for child_note in hierarchy[note]:
                    direct_replies.add(child_note)
        self.unrec_noteids.union(direct_replies)

    def get_ppr_recommendations(self, ppr_values, eval_list, read_set, k = 10, ratio_read=0.3):
        num_read_recs = int(ratio_read*k)
        num_unread_recs = k - num_read_recs
        if len(eval_list) > k and num_read_recs!=0:
            never_seen_posts = eval_list - read_set
            ppr_values_never_seen = {node_id:ppr for node_id,ppr in ppr_values.items() if node_id in never_seen_posts}
            ppr_values_read = {node_id:ppr for node_id,ppr in ppr_values.items() if node_id in read_set}
            shortage_ns = num_unread_recs - len(ppr_values_never_seen)
            shortage_r = num_read_recs - len(ppr_values_read)
            if shortage_ns<0 and shortage_r<0:                
                topk_never_seen = utils.topn_from_dict(ppr_values_never_seen, num_unread_recs)
                topk_read = utils.topn_from_dict(ppr_values_read, num_read_recs)
            elif shortage_ns>0 and shortage_r<=0:
                topk_never_seen = utils.topn_from_dict(ppr_values_never_seen, len(ppr_values_never_seen))
                topk_read = utils.topn_from_dict(ppr_values_read, k-len(ppr_values_never_seen))
            elif shortage_r>0 and shortage_ns<=0:
                topk_never_seen = utils.topn_from_dict(ppr_values_never_seen, k-len(ppr_values_read))
                topk_read = utils.topn_from_dict(ppr_values_read, len(ppr_values_read))
            else:
                topk_never_seen = utils.topn_from_dict(ppr_values_never_seen, len(ppr_values_never_seen))
                topk_read = utils.topn_from_dict(ppr_values_read, k-len(ppr_values_never_seen))
                
            return list(topk_never_seen+topk_read)

        elif num_read_recs==0:
            never_seen_posts = eval_list - read_set
            ppr_values_never_seen = {node_id:ppr for node_id,ppr in ppr_values.items() if node_id in never_seen_posts}
            topk_never_seen = utils.topn_from_dict(ppr_values_never_seen, num_unread_recs)

            return list(topk_never_seen)

        else:
            ppr_values = {node_id:ppr for node_id,ppr in ppr_values.items() if node_id in eval_list}
            topk = utils.topn_from_dict(ppr_values, k)
            
            return list(topk)