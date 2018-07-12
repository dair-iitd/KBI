import numpy as np
import random
import sys
import theano 
import theano.tensor as T

def adder_eval():
    entityPairs  = T.fmatrix()
    entities  = T.fmatrix()
    relations_DM = T.fmatrix()
    relations_MF = T.fmatrix()
    testData_DM = T.imatrix()
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


    oov_flag_e1_DM = T.ivector()
    oov_flag_e2_DM = T.ivector()
    oov_flags_MF =T.ivector()

    def MF_fn(testPoint_DM, testPoint_MF, i, oov_flag_e1, oov_flag_e2, oov_flag, entityPairs, entities, relations_DM, relations_MF, entityPair_oov_embedding, entity_oov_embedding, allowed_entityPair, set1_e2, set2_e2, set3_e2):

        r_idx = testPoint_DM[1]

        e1_idx = testPoint_DM[0]
        e2_idx = testPoint_DM[2]
        ep_idx = testPoint_MF[1]

        r_embedding_MF = relations_MF[r_idx]
        r_embedding_DM = relations_DM[r_idx]

        e1_fact_embedding = T.switch(oov_flag_e1, entity_oov_embedding, entities[e1_idx])
        e2_fact_embedding = T.switch(oov_flag_e2, entity_oov_embedding, entities[e2_idx])
        ep_fact_embedding = T.switch(oov_flag, entityPair_oov_embedding, entityPairs[ep_idx])

        score_testPoint = T.dot(r_embedding_MF, ep_fact_embedding) + T.dot(r_embedding_DM, e1_fact_embedding*e2_fact_embedding)

        score_DM_set1 = T.dot(e1_fact_embedding*entities[set1_e2[i]], r_embedding_DM)
        score_DM_set2 = T.dot(e1_fact_embedding*entities[set2_e2[i]], r_embedding_DM)
        score_DM_set3 = T.dot(e1_fact_embedding*entity_oov_embedding, r_embedding_DM)

        ep_embeddings_set1 = entityPairs[allowed_entityPair[i]]
        oov_score_MF = T.dot(r_embedding_MF, entityPair_oov_embedding)


        scores_set1 = score_DM_set1 + T.dot(ep_embeddings_set1, r_embedding_MF)
        scores_set2 = score_DM_set2 + oov_score_MF
        scores_set3 = score_DM_set3 + oov_score_MF


        rank = 1 + T.sum(scores_set1 > score_testPoint) + T.sum(scores_set2 > score_testPoint)
        same = T.sum(T.eq(scores_set1, score_testPoint)) + T.sum(T.eq(scores_set2,score_testPoint))

        rank += same/2.0
        oov_comparison = score_testPoint < scores_set3
        rank = T.switch(oov_comparison, rank + set3_e2[i], rank)
        rank = T.switch(oov_flag_e2, rank + (set3_e2[i]/2.0), rank)

        return rank


    ranks, ignore = theano.scan(MF_fn, non_sequences = [entityPairs, entities,relations_DM, relations_MF, entityPair_oov_embedding, entity_oov_embedding, allowedEP_MF, set1_e2, set2_e2, set3_e2], sequences = [testData_DM, testData_MF, theano.tensor.arange(testData_DM.shape[0]), oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF])
    f = theano.function([entityPairs, entities, relations_DM, relations_MF, entityPair_oov_embedding, entity_oov_embedding, testData_DM, testData_MF, allowedEP_MF, set1_e2, set2_e2, oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF, set3_e2], ranks,allow_input_downcast=True)

    return f


def get_complex_score(x, y, relation_vectors):
    a00 = x[0]*y[0]
    a01 = x[0]*y[1]
    a11 = x[1]*y[1]
    a10 = x[1]*y[0]

    r1 = T.dot(a00, relation_vectors[0])
    r2 = T.dot(a11, relation_vectors[0])
    r3 = T.dot(a01, relation_vectors[1])
    r4 = T.dot(a10, relation_vectors[1])

    result = r1 + r2 + r3 - r4
    return result

def adder_eval_complex():
    entityPairs  = T.fmatrix()
    entities_i  = T.fmatrix()
    entities_r  = T.fmatrix()
    relations_i = T.fmatrix()
    relations_r = T.fmatrix()
    relations_MF = T.fmatrix()
    testData_DM = T.imatrix()
    testData_MF = T.imatrix()
    entity_oov_embedding_i = T.fvector()
    entity_oov_embedding_r = T.fvector()
    entityPair_oov_embedding = T.fvector()
    '''
        for a given (e1, ?): we can partition the filtered candidate e2s into:

        1) e2s such that (e1,e2) is trained -> allowedEP_MF
    '''
    allowedEP_MF = theano.typed_list.TypedListType(T.ivector)()
    set1_e2  = theano.typed_list.TypedListType(T.ivector)()
    set2_e2 = theano.typed_list.TypedListType(T.ivector)()
    set3_e2= T.ivector()

    oov_flag_e1_DM = T.ivector()
    oov_flag_e2_DM = T.ivector()
    oov_flags_MF =T.ivector()
    

    def MF_fn(testPoint_DM, testPoint_MF, i, oov_flag_e1, oov_flag_e2, oov_flag, entityPairs, entities_r, entities_i, relations_r, relations_i, relations_MF, entityPair_oov_embedding, entity_oov_embedding_r, entity_oov_embedding_i, allowed_entityPair, set1_e2, set2_e2, set3_e2):

        r_idx = testPoint_DM[1]

        e1_idx = testPoint_DM[0]
        e2_idx = testPoint_DM[2]
        ep_idx = testPoint_MF[1]

        r_embedding_MF = relations_MF[r_idx]
        ep_fact_embedding = T.switch(oov_flag, entityPair_oov_embedding, entityPairs[ep_idx])

        score_testPoint = T.dot(r_embedding_MF, ep_fact_embedding) 

        e1_fact_embedding_real = T.switch(oov_flag_e1, entity_oov_embedding_r, entities_r[e1_idx])
        e1_fact_embedding_im = T.switch(oov_flag_e1, entity_oov_embedding_i, entities_i[e1_idx])

        score_DM_set1 = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entities_r[set1_e2[i]], entities_i[set1_e2[i]]],
                                [relations_r[r_idx], relations_i[r_idx]])
        score_DM_set2 = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entities_r[set2_e2[i]], entities_i[set2_e2[i]]],
                                [relations_r[r_idx], relations_i[r_idx]])
        score_DM_set3 = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entity_oov_embedding_r, entity_oov_embedding_i],
                                [relations_r[r_idx], relations_i[r_idx]])

        #scores = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entities_r, entities_i],
                                #[relations_r[testPoint[1]], relations_i[testPoint[1]]])
        score_testPoint = score_testPoint + get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], 
                                                              [entities_r[e2_idx], entities_i[e2_idx]],
                                                              [relations_r[r_idx], relations_i[r_idx]])#scores[testPoint[2]]

        ep_embeddings_set1 = entityPairs[allowed_entityPair[i]]
        oov_score_MF = T.dot(r_embedding_MF, entityPair_oov_embedding)


        scores_set1 = score_DM_set1 + T.dot(ep_embeddings_set1, r_embedding_MF)
        scores_set2 = score_DM_set2 + oov_score_MF
        scores_set3 = score_DM_set3 + oov_score_MF


        rank = 1 + T.sum(scores_set1 > score_testPoint) + T.sum(scores_set2 > score_testPoint)
        same = T.sum(T.eq(scores_set1, score_testPoint)) + T.sum(T.eq(scores_set2,score_testPoint))

        rank += same/2.0
        oov_comparison = score_testPoint < scores_set3
        rank = T.switch(oov_comparison, rank + set3_e2[i], rank)
        rank = T.switch(oov_flag_e2, rank + (set3_e2[i]/2.0), rank)          
        
        return rank

                                                            
    ranks, ignore = theano.scan(MF_fn, non_sequences = [entityPairs, entities_r, entities_i, relations_r, relations_i, relations_MF, entityPair_oov_embedding, entity_oov_embedding_r, entity_oov_embedding_i, allowedEP_MF, set1_e2, set2_e2, set3_e2], sequences = [testData_DM, testData_MF, theano.tensor.arange(testData_DM.shape[0]), oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF])                                                        
    f = theano.function([entityPairs, entities_r, entities_i, relations_r, relations_i, relations_MF, entityPair_oov_embedding, entity_oov_embedding_r, entity_oov_embedding_i, testData_DM, testData_MF, allowedEP_MF, set1_e2, set2_e2, oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF, set3_e2], ranks,allow_input_downcast=True)
    return f



def get_model_weights(model):
    print("evaluating.")

    for layer in model.layers:
        if layer.name == 'entity_embeddings_real':
            e_real2 = layer.get_weights()[0];
        elif layer.name == 'entity_embeddings_im':
            e_im2 = layer.get_weights()[0];
        elif layer.name == 'relation_embeddings_real':
            r_real2 = layer.get_weights()[0];
        elif layer.name == 'relation_embeddings_im':
            r_im2 = layer.get_weights()[0];
    return e_real2, e_im2, r_real2, r_im2
'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_counts[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def score_adder(model, opts, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM, eval_func):

    nnet_layers = {} 
    print("evaluating.")
    trained_layers = ["entity_embeddings_MF", "relation_embeddings", "entity_embeddings_DM", "relation_embeddings_MF", "relation_embeddings_DM"]

    for layer in model.layers:
        if layer.name == 'entity_embeddings_MF':
            entityPair_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_MF':
            relation_weights_MF = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_DM':
            relation_weights_DM = layer.get_weights()[0]

        if layer.name in trained_layers:
            print(layer.name + " shape = " + str(layer.get_weights()[0].shape))

    if opts.use_complex:
        e_real, e_im, r_real, r_im = get_model_weights(model)
        oov_embedding_e_real = e_real[-1]
        oov_embedding_e_im = e_im[-1]
    else:
        oov_embedding_e = entity_weights[-1]
    oov_embedding_ep = entityPair_weights[-1]


    #Not upgraded
    if opts.shared_r:
        print("relations shared")
        relation_weights_MF = relation_weights_DM = relation_weights 
    #

    if opts.use_complex:
        ranks = eval_func(entityPair_weights, e_real, e_im, r_real, r_im, relation_weights_MF, oov_embedding_ep, oov_embedding_e_real,oov_embedding_e_im, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM)
    else:
        ranks = eval_func(entityPair_weights, entity_weights, relation_weights_DM, relation_weights_MF, oov_embedding_ep, oov_embedding_e, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM) 


    ranks_oov = []
    ranks_others =[]
    for i in xrange(len(ranks)):
        if oov_flags_MF[i]:
            ranks_oov.append(ranks[i])
        else:
            ranks_others.append(ranks[i])    


    ranks_oov = np.array(ranks_oov)
    ranks_others = np.array(ranks_others)
    print("NON OOVS: MRR: %5.4f, HITS: %5.4f" %(np.mean(1.0/ranks_others), np.mean(ranks_others <= 10.0)))
    print np.mean(1.0/ranks_others), np.mean(ranks_others <= 10.0), np.mean(ranks_others <= 1.0), np.mean(ranks_others)
    print("OOVS: MRR: %5.4f, HITS: %5.4f" %(np.mean(1.0/ranks_oov), np.mean(ranks_oov <= 10.0)))
    print np.mean(1.0/ranks_oov), np.mean(ranks_oov <= 10.0), np.mean(ranks_oov <= 1.0),np.mean(ranks_oov)

    print "Net:: ",np.mean(1.0/ranks), np.mean(ranks <= 10.0),np.mean(ranks <= 1.0), np.mean(ranks)

    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)
