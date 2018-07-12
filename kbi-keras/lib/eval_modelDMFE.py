import numpy as np
import random
import sys
import theano 
import theano.tensor as T


def model_eval_DMFE():
    entities_E  = T.fmatrix()
    relations_E = T.fmatrix()

    entities_DM = T.fmatrix()
    relations_DM = T.fmatrix()

    entities_MF  = T.fmatrix()    
    relations_MF = T.fmatrix()

    testData_E = T.imatrix()
    testData_MF = T.imatrix()

    entity_oov_embedding_E = T.fvector()
    entity_oov_embedding_DM = T.fvector()
    entityPair_oov_embedding = T.fvector()
    '''
        for a given (e1, ?): we can partition the filtered candidate e2s into:

        1) e2s such that (e1,e2) is trained -> allowedEP_MF
    '''
    allowedEP_MF = theano.typed_list.TypedListType(T.ivector)()
    set1_e2  = theano.typed_list.TypedListType(T.ivector)()
    set2_e2 = theano.typed_list.TypedListType(T.ivector)()
    set3_e2= T.ivector()
    
    oov_flags_e1 = T.ivector()
    oov_flags_e2 = T.ivector()
    oov_flags_MF =T.ivector()
    

    def MF_fn(testPoint_E, testPoint_MF, i, oov_flag_e1, oov_flag_e2, oov_flag, entities_MF, relations_MF, entities_DM, relations_DM, entities_E, relations_E, entityPair_oov_embedding, entity_oov_embedding_DM, entity_oov_embedding_E, allowed_entityPair, set1_e2, set2_e2, set3_e2):
        r_idx = testPoint_E[1]
        e2_idx = testPoint_E[2]
        e1_idx = testPoint_E[0]
        ep_idx = testPoint_MF[1]

        # get r embeddings for all 3 models
        r_embedding_E = relations_E[r_idx]
        r_embedding_MF = relations_MF[r_idx]
        r_embedding_DM = relations_DM[r_idx]

        # get e1 and e2 DM embeddings
        e1_fact_embedding_DM = T.switch(oov_flag_e1, entity_oov_embedding_DM, entities_DM[e1_idx])
        e2_fact_embedding_DM = T.switch(oov_flag_e2, entity_oov_embedding_DM, entities_DM[e2_idx])

        # get e2 embeddings for E
        e2_fact_embedding_E = T.switch(oov_flag_e2, entity_oov_embedding_E, entities_E[e2_idx])

        # get (e1,e2) embeddings for MF
        ep_fact_embedding = T.switch(oov_flag, entityPair_oov_embedding, entities_MF[ep_idx])
        
        # score testpoint = r_E*e2 + r_DM*(e1,e2) + r_DM*(e1 o e2)
        score_testPoint = T.dot(r_embedding_MF, ep_fact_embedding) + T.dot(r_embedding_E, e2_fact_embedding_E) + T.dot(r_embedding_DM, e1_fact_embedding_DM*e2_fact_embedding_DM)

        score_E_set1 = T.dot(entities_E[set1_e2[i]], r_embedding_E)
        score_E_set2 = T.dot(entities_E[set2_e2[i]], r_embedding_E)
        score_E_set3 = T.dot(entity_oov_embedding_E, r_embedding_E)

        score_DM_set1 = T.dot(e1_fact_embedding_DM*entities_DM[set1_e2[i]], r_embedding_DM)
        score_DM_set2 = T.dot(e1_fact_embedding_DM*entities_DM[set2_e2[i]], r_embedding_DM)
        score_DM_set3 = T.dot(e1_fact_embedding_DM*entity_oov_embedding_DM, r_embedding_DM)


        score_MF_set1 = T.dot(entities_MF[allowed_entityPair[i]], r_embedding_MF)
        oov_score_MF = T.dot(r_embedding_MF, entityPair_oov_embedding)

        scores_set1 = score_DM_set1 + score_E_set1 + score_MF_set1
        scores_set2 = score_DM_set2 + score_E_set2 + oov_score_MF
        scores_set3 = score_DM_set3 + score_E_set3 + oov_score_MF


        rank = 1 + T.sum(scores_set1 > score_testPoint) + T.sum(scores_set2 > score_testPoint)
        same = T.sum(T.eq(scores_set1, score_testPoint)) + T.sum(T.eq(scores_set2,score_testPoint))

        rank += same/2.0
        oov_comparison = score_testPoint < scores_set3
        rank = T.switch(oov_comparison, rank + set3_e2[i], rank)
        rank = T.switch(oov_flag_e2, rank + (set3_e2[i]/2.0), rank)          
        
        return rank

                                                            
    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities_MF, relations_MF, entities_DM, relations_DM, entities_E, relations_E, entityPair_oov_embedding, entity_oov_embedding_DM, entity_oov_embedding_E,  allowedEP_MF, set1_e2, set2_e2, set3_e2], sequences = [testData_E, testData_MF, theano.tensor.arange(testData_E.shape[0]), oov_flags_e1, oov_flags_e2, oov_flags_MF])                                                        

    f = theano.function([entities_DM, relations_DM, entities_E, relations_E, entities_MF, relations_MF, entityPair_oov_embedding, entity_oov_embedding_E, entity_oov_embedding_DM, testData_E, testData_MF, allowedEP_MF, set1_e2, set2_e2, oov_flags_e1, oov_flags_e2, oov_flags_MF, set3_e2], ranks,allow_input_downcast=True)
    
    return f


'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_counts[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def score_DMFE(model, opts, testData_E, testData_MF, allowedEP, allowedEP_e2, notAllowedEP_e2, oov_flags_e1, oov_flags_e2, oov_flags_MF, oov_e2_count, eval_func):

    nnet_layers = {} 
    print("evaluating.")
    for layer in model.layers:
        # get all MF embeddings
        if layer.name == 'entity_embeddings_MF':
            entityPair_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights_MF = layer.get_weights()[0]
        # get all E embeddings
        elif layer.name == 'entity_embeddings':
            entity_weights_E = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights_E = layer.get_weights()[0]
        #get all DM embeddings
        elif layer.name == 'entity_embeddings_DM':
            entity_weights_DM = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_DM':
            relation_weights_DM = layer.get_weights()[0]

    oov_embedding_ep = entityPair_weights[-1]
    oov_embedding_e = entity_weights_E[-1] 
    oov_embedding_DM = entity_weights_DM[-1] 

    ranks = eval_func(entity_weights_DM, relation_weights_DM,entity_weights_E, relation_weights_E, entityPair_weights, relation_weights_MF,oov_embedding_ep, oov_embedding_e, oov_embedding_DM, testData_E, testData_MF, allowedEP, allowedEP_e2, notAllowedEP_e2, oov_flags_e1, oov_flags_e2, oov_flags_MF, oov_e2_count) 

    mrr = np.mean(1.0/ranks)
    hits = np.mean(ranks <= 10.0)
    print mrr, hits, np.mean(ranks<=1), np.mean(ranks)
    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)

def model_eval_DME():
    entities_E  = T.fmatrix(); relations_E = T.fmatrix()
    entities_DM = T.fmatrix(); relations_DM = T.fmatrix()
    testData_E = T.imatrix()
    entity_oov_embedding_E = T.fvector(); entity_oov_embedding_DM = T.fvector() 
    oov_flags_e1 = T.ivector(); oov_flags_e2 = T.ivector()
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    oov_e2 = T.ivector()   
    def MF_fn(testPoint_E, i, oov_flag_e1, oov_flag_e2, entities_DM, relations_DM, entities_E, relations_E,  entity_oov_embedding_E, entity_oov_embedding_DM, seen_e2, oov_e2):
        r_idx = testPoint_E[1]; e2_idx = testPoint_E[2]; e1_idx = testPoint_E[0]

        r_embedding_E = relations_E[r_idx]; r_embedding_DM = relations_DM[r_idx]

        e1_fact_embedding_DM = T.switch(oov_flag_e1, entity_oov_embedding_DM, entities_DM[e1_idx]); 
        e2_fact_embedding_DM = T.switch(oov_flag_e2, entity_oov_embedding_DM, entities_DM[e2_idx])
        e2_fact_embedding_E = T.switch(oov_flag_e2, entity_oov_embedding_E, entities_E[e2_idx])

        score_testPoint = T.dot(r_embedding_E, e2_fact_embedding_E) + T.dot(r_embedding_DM, e1_fact_embedding_DM*e2_fact_embedding_DM)
        
        e1_cross_e2_all  = e1_fact_embedding_DM*entities_DM
        
        scores_E = T.dot(entities_E, r_embedding_E) 
        scores_DM   = T.dot(e1_cross_e2_all, r_embedding_DM)
        scores = scores_E + scores_DM

        rank1 = 1 + T.sum(scores > score_testPoint)
        rank = rank1 - T.sum(scores[seen_e2[i]] > score_testPoint)

        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(r_embedding_DM, e1_fact_embedding_DM * entity_oov_embedding_DM) + T.dot(r_embedding_E, entity_oov_embedding_E)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        rank = T.switch(oov_comparison, rank, rank + oov_e2[i])
        rank = T.switch(oov_flag_e2, rank + oov_e2[i]/2.0, rank)
        return rank, oov_comparison, score_testPoint,rank1
        ###

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities_DM, relations_DM, entities_E, relations_E, entity_oov_embedding_E, entity_oov_embedding_DM, seen_e2, oov_e2], sequences = [testData_E, theano.tensor.arange(testData_E.shape[0]), oov_flags_e1, oov_flags_e2])

    f = theano.function([entities_DM, relations_DM, entities_E, relations_E, entity_oov_embedding_E, entity_oov_embedding_DM, seen_e2, oov_e2, testData_E, oov_flags_e1, oov_flags_e2], ranks,allow_input_downcast=True)

    return f


def score_DME(model, opts, testData_E, oov_flags_e1, oov_flags_e2, oov_e2_count, seen_e2, eval_func):
    nnet_layers = {}
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights_E = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights_E = layer.get_weights()[0]
        #get all DM embeddings
        elif layer.name == 'entity_embeddings_DM':
            entity_weights_DM = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_DM':
            relation_weights_DM = layer.get_weights()[0]

    oov_embedding_e = entity_weights_E[-1]
    oov_embedding_DM = entity_weights_DM[-1]

    ranks = eval_func(entity_weights_DM, relation_weights_DM,entity_weights_E, relation_weights_E, oov_embedding_e, oov_embedding_DM, seen_e2, oov_e2_count, testData_E, oov_flags_e1, oov_flags_e2)

    mrr = np.mean(1.0/ranks[0])
    hits = np.mean(ranks[0] <= 10.0)
    print mrr, hits, np.mean(ranks[0]<=1), np.mean(ranks[0])
    print("FILTERED:: MRR:%3.4f HITS@10::%3.4f HITS@1:%3.4f MR:%3.4f" %(np.mean(1.0/ranks[0]),np.mean(ranks[0]<=10.0),np.mean(ranks[0]<=1.0),np.mean(ranks[0])))
    print("RAW     :: MRR:%3.4f HITS@10::%3.4f HITS@1:%3.4f MR:%3.4f" %(np.mean(1.0/ranks[-1]),np.mean(ranks[-1]<=10.0),np.mean(ranks[-1]<=1.0),np.mean(ranks[-1])))
    return mrr,hits

