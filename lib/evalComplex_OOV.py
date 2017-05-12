import numpy as np
import random
import theano
import theano.typed_list
import theano.tensor as T
from collections import defaultdict as ddict, Counter
from evalTensor_OOV import *
import joblib
import sys, os


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
    
def complex_run():
    e_real = T.dmatrix()
    e_im = T.dmatrix()
    r_real = T.dmatrix()
    r_im = T.dmatrix()
    testData = T.imatrix()

    def complex_fn(testPoint, e_real, e_im, r_real, r_im):
        e1_curr_real = e_real[testPoint[0]] 
        e2_curr_real = e_real[testPoint[2]]
        e1_curr_im = e_im[testPoint[0]]
        e2_curr_im = e_im[testPoint[2]]
        r1_curr_real = r_real[testPoint[1]]
        r1_curr_im = r_real[testPoint[1]]

        return get_complex_score([e1_curr_real, e1_curr_im], [e2_curr_real, e2_curr_im], [r1_curr_real, r1_curr_im])

    scoredm, ignore = theano.scan(complex_fn, non_sequences = [e_real, e_im, r_real, r_im], sequences = [testData])
    f = theano.function([e_real, e_im,r_real, r_im, testData], scoredm,allow_input_downcast=True)

    return f

'''
    TODO
'''
def complex_get_scores():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0

    def distMult_fn(testPoint, oov_flag1, entities, relations):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e1_cross_e2_all  = e1_fact_embedding*entities
        scores   = T.dot(e1_cross_e2_all, relations[testPoint[1]])
        return scores

    scoremat, ignore = theano.scan(distMult_fn, non_sequences = [entities,relations], sequences = [testData, oov_flags_1])
    f = theano.function([entities,relations,testData, oov_flags_1, oov_embedding], scoremat,allow_input_downcast=True)

    return f



def complex_oovEval():
    entities_real  = T.dmatrix()
    entities_im = T.dmatrix()
    relations_real = T.dmatrix()
    relations_im = T.dmatrix()
    oov_embedding_real = T.dvector()
    oov_embedding_im = T.dvector()

    testData = T.imatrix()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()

    def complex_fn(testPoint, oov_flag1, oov_flag2, i, entities_real, entities_im, relations_real, relations_im, seen_e2):
        e1_fact_embedding_real = T.switch(oov_flag1, oov_embedding_real, entities_real[testPoint[0]])
        e1_fact_embedding_im = T.switch(oov_flag1, oov_embedding_im, entities_im[testPoint[0]])
        e2_fact_embedding_real = T.switch(oov_flag2, oov_embedding_real, entities_real[testPoint[2]])
        e2_fact_embedding_im = T.switch(oov_flag2, oov_embedding_im, entities_im[testPoint[2]])

        scores = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entities_real, entities_im], 
                                [relations_real[testPoint[1]], relations_im[testPoint[1]]])
        score_testPoint = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [e2_fact_embedding_real, e2_fact_embedding_im], 
                                [relations_real[testPoint[1]], relations_im[testPoint[1]]])

        rank = 1 + T.sum(scores > score_testPoint)-T.sum(scores[seen_e2[i]] > score_testPoint)
        #same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [oov_embedding_real, oov_embedding_im],
                                      [relations_real[testPoint[1]], relations_im[testPoint[1]]])
        oov_comparison = score_testPoint < score_oov
        #rank += same/2.0
    
        return rank, oov_comparison, score_testPoint

    ranks, ignore = theano.scan(complex_fn, non_sequences = [entities_real, entities_im, relations_real, relations_im, seen_e2], 
                                sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])

    f = theano.function([entities_real, entities_im, relations_real, relations_im, testData, seen_e2, oov_flags_1, 
                         oov_flags_2, oov_embedding_real, oov_embedding_im], ranks,allow_input_downcast=True)    

    return f



def score_complex(model, opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval):
    for layer in model.layers:
        if layer.name == 'entity_embeddings_real': 
            e_real = layer.get_weights()[0]
        elif layer.name == 'entity_embeddings_im': 
            e_im = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_real': 
            r_real = layer.get_weights()[0]
        elif layer.name.startswith('relation_embeddings_im'):
            r_im = layer.get_weights()[0]
    
    
    return score_complex_helper( [e_real, e_im, r_real, r_im] , opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval)


def score_complex_helper(model_weights, opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval):

    e_real, e_im, r_real, r_im = model_weights

    oov_embedding_real = e_real[-1]
    oov_embedding_im = e_im[-1]

    if os.path.exists("ranks_FB15K_complex.joblib"):
        ranks = joblib.load("ranks_FB15K_complex.joblib")

    else:
        ranks = complex_eval(e_real, e_im, r_real, r_im, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding_real, oov_embedding_im)
        for i, comparison_bit in enumerate(ranks[1]):
            if (comparison_bit):
                ranks[0][i] += oov_e2[i]
        joblib.dump(ranks, "ranks_FB15K_complex.joblib")


    def get_preds(rank_id_pairs):
        test_curr = []; seen_e2_curr = []; oov_flags_1_curr = []

        for (rank, i) in rank_id_pairs:
            test_curr.append(testData[i]); 
            seen_e2_curr.append(seen_e2[i]); 
            oov_flags_1_curr.append(oov_flags_1[i]); 

        score_matrix = complex_get_scores(entity_weights, relation_weights, test_curr, oov_flags_1_curr, oov_embedding)
        preds = []

        for i, score_vector in enumerate(score_matrix):
            e1, r, e2 = test_curr[i]
            best_pred = get_best(score_vector, seen_e2_curr[i], e2)
            preds.append(best_pred)

        return preds

    def get_inverse_scores(inv_relation_arr):
        test_curr = []
        for i, test_point in enumerate(testData):
            e1, r, e2 = test_point
            inv_score, inv_r = inv_relation_arr[i]
            if (inv_r is not  None):
                test_curr.append(np.array([e2, inv_r, e1]))
            else:
                test_curr.append(np.array([e1, r, e2]))


        score_distMult = complex_run(entity_weights, relation_weights, test_curr)
        return score_distMult



    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)
