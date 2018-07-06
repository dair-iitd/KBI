import numpy as np
import random
import sys
import theano 
import theano.tensor as T


def model_eval_FE():
    entityPairs  = T.fmatrix()
    entities  = T.fmatrix()
    relations_E = T.fmatrix()
    relations_MF = T.fmatrix()
    testData_E = T.imatrix()
    testData_MF = T.imatrix()
    entity_oov_embedding = T.fvector()
    entityPair_oov_embedding = T.fvector()
    '''
        for a given (e1, ?): we can partition the filtered candidate e2s into:

        1) e2s such that (e1,e2) is trained -> allowedEP_MF
    '''
    allowedEP_MF = theano.typed_list.TypedListType(T.ivector)()
    set1_e2  = theano.typed_list.TypedListType(T.ivector)()
    set2_e2 = theano.typed_list.TypedListType(T.ivector)()
    set3_e2= T.ivector()

    
    oov_flags_e2 = T.ivector()
    oov_flags_MF =T.ivector()
    

    def MF_fn(testPoint_E, testPoint_MF, i, oov_flag_e2, oov_flag, entityPairs, entities, relations_E, relations_MF, entityPair_oov_embedding, entity_oov_embedding, allowed_entityPair, set1_e2, set2_e2, set3_e2):
        r_idx = testPoint_E[1]
        e2_idx = testPoint_E[2]
        ep_idx = testPoint_MF[1]

        r_embedding_E = relations_E[r_idx]
        r_embedding_MF = relations_MF[r_idx]

        e2_fact_embedding = T.switch(oov_flag_e2, entity_oov_embedding, entities[e2_idx])
        ep_fact_embedding = T.switch(oov_flag, entityPair_oov_embedding, entityPairs[ep_idx])
        

        score_testPoint = T.dot(r_embedding_MF, ep_fact_embedding) + T.dot(r_embedding_E, e2_fact_embedding)
        score_E_set1 = T.dot(entities[set1_e2[i]], r_embedding_E)
        score_E_set2 = T.dot(entities[set2_e2[i]], r_embedding_E)
        score_E_set3 = T.dot(entity_oov_embedding, r_embedding_E)

        ep_embeddings_set1 = entityPairs[allowed_entityPair[i]]
        oov_score_MF = T.dot(r_embedding_MF, entityPair_oov_embedding)

        scores_set1 = score_E_set1 + T.dot(ep_embeddings_set1, r_embedding_MF)
        scores_set2 = score_E_set2 + oov_score_MF
        scores_set3 = score_E_set3 + oov_score_MF


        rank = 1 + T.sum(scores_set1 > score_testPoint) + T.sum(scores_set2 > score_testPoint)
        same = T.sum(T.eq(scores_set1, score_testPoint)) + T.sum(T.eq(scores_set2,score_testPoint))

        rank += same/2.0
        oov_comparison = score_testPoint < scores_set3
        rank = T.switch(oov_comparison, rank + set3_e2[i], rank)
        rank = T.switch(oov_flag_e2, rank + (set3_e2[i]/2.0), rank)          
        
        return rank

                                                            
    ranks, ignore = theano.scan(MF_fn, non_sequences = [entityPairs, entities,relations_E, relations_MF, entityPair_oov_embedding, entity_oov_embedding, allowedEP_MF, set1_e2, set2_e2, set3_e2], sequences = [testData_E, testData_MF, theano.tensor.arange(testData_E.shape[0]), oov_flags_e2, oov_flags_MF])                                                        
    f = theano.function([entityPairs, entities, relations_E, relations_MF, entityPair_oov_embedding, entity_oov_embedding, testData_E, testData_MF, allowedEP_MF, set1_e2, set2_e2, oov_flags_e2, oov_flags_MF, set3_e2], ranks,allow_input_downcast=True)
    
    return f


'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_counts[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def score_FE(model, opts, testData_E, testData_MF, allowedEP, allowedEP_e2, notAllowedEP_e2, oov_flags_e2, oov_flags_MF, oov_e2_count, eval_func):

    nnet_layers = {} 
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings_MF':
            entityPair_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights_MF = layer.get_weights()[0]
        elif layer.name == 'entity_embeddings':
            entity_weights_E = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights_E = layer.get_weights()[0]

    oov_embedding_ep = entityPair_weights[-1]
    oov_embedding_e = entity_weights_E[-1] 


    ranks = eval_func(entityPair_weights, entity_weights_E, relation_weights_E, relation_weights_MF,oov_embedding_ep, oov_embedding_e, testData_E, testData_MF, allowedEP, allowedEP_e2, notAllowedEP_e2, oov_flags_e2, oov_flags_MF, oov_e2_count) 

    mrr = np.mean(1.0/ranks)
    hits = np.mean(ranks <= 10.0)

    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)
