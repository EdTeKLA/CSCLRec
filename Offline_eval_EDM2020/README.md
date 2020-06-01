## Code for CSCLRec and its competitors  
- CSCLRec: CSCLRec.py  
- CoPPR: CoPPR.py  
- PPR: PurePPR.py  
- KCB and SCB: CBF.py  
- MCF: MCF.py  
- PPL and RND: in OfflineEvaluator.py (see run_popularity_eval() and run_random_eval())

### Hyperparameters tuned for each recommender

|  Method |                                                                            Hyperparameters                                                                            | Values                                       |
|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------------------------------------------|
| CSCLRec | Temporal ratio applied to older posts (= 1-temporal decay),    Number of peer learners for each learner to be connected in the PPR graph,    Damping factor for the PPR graph | [0.7,0.75,0.8],  [3,5,7,10],  [0.8,0.85,0.9] |
|  CoPPR  | Temporal ratio applied to older posts (= 1-temporal decay),    Number of peer learners for each learner to be connected in the PPR graph,    Damping factor for the PPR graph       | [0.7,0.75,0.8],  [3,5,7,10],  [0.8,0.85,0.9] |
|   PPR   | Damping factor for the PPR graph                                                                                                                                      | [0.75, 0.8, 0.85]                            |
|   MCF   | Confidence scaling term (the alpha term as defined in http://yifanhu.net/PUB/cf.pdf)                                                                                           | [15,20,25,30]                                |
|   KCB   | Word vector dimension size, top 1/n content words being used as keywords to represent the post                                                                        | [10,15,20],  [3,5]                           |

### Example: The list of hyperparamter values chosen after tuning for the course LA

| Method/Week# |       2      |       3      |       4      |       5      |
|:------------:|:------------:|:------------:|:------------:|:------------:|
|    CSCLRec   | 0.75, 7, 0.9 | 0.7, 10, 0.9 | 0.7, 10, 0.8 | 0.8, 10, 0.9 |
|     CoPPR    | 0.75, 7, 0.9 | 0.7, 10, 0.9 | 0.7, 10, 0.9 | 0.7, 10, 0.9 |
|      PPR     |     0.85     |     0.85     |     0.85     |     0.85     |
|      MCF     |      15      |      30      |      20      |      25      |
|      KCB     |     15, 3    |     15, 5    |     20, 5    |     10, 5    |

The week# indicates the week where this set of hyperparamters is applied to generate recommendations.


Referencing github repos:
1. https://github.com/networkx/networkx  
2. https://github.com/RaRe-Technologies/gensim  
3. https://github.com/benfred/implicit  
4. https://github.com/peter3125/sentence2vec