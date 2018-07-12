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
        #e2_fact_embedding_real = T.switch(oov_flag2, oov_embedding_real, entities_real[testPoint[2]])
        #e2_fact_embedding_im = T.switch(oov_flag2, oov_embedding_im, entities_im[testPoint[2]])

        scores = get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [entities_real, entities_im], 
                                [relations_real[testPoint[1]], relations_im[testPoint[1]]])
        score_testPoint = scores[testPoint[2]]#get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [e2_fact_embedding_real, e2_fact_embedding_im], 
                                #[relations_real[testPoint[1]], relations_im[testPoint[1]]])

        #if 0:
        #rank1 = 1 + T.sum(scores[seen_e2[i]] > score_testPoint)
        #rank = rank1
        #else:
        rank1= 1 + T.sum(scores[:-1] > score_testPoint)#dec29'17 - do not compare with oov embed at this stage
        rank = rank1 - T.sum(scores[seen_e2[i]] > score_testPoint)

        #same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = scores[-1]#get_complex_score([e1_fact_embedding_real, e1_fact_embedding_im], [oov_embedding_real, oov_embedding_im],
                                      #[relations_real[testPoint[1]], relations_im[testPoint[1]]])
        oov_comparison = score_testPoint < score_oov
        #rank += same/2.0
    
        return rank, oov_comparison, score_testPoint, scores, rank1

    ranks, ignore = theano.scan(complex_fn, non_sequences = [entities_real, entities_im, relations_real, relations_im, seen_e2], 
                                sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])

    f = theano.function([entities_real, entities_im, relations_real, relations_im, testData, seen_e2, oov_flags_1, 
                         oov_flags_2, oov_embedding_real, oov_embedding_im], ranks,allow_input_downcast=True)    

    return f



def score_complex(model, opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval):
    for layer in model.layers:
        if layer.name == 'entity_embeddings_real': 
            e_real = layer.get_weights()[0];print layer.name
        elif layer.name == 'entity_embeddings_im': 
            e_im = layer.get_weights()[0];print layer.name
        elif layer.name == 'relation_embeddings_real': 
            r_real = layer.get_weights()[0];print layer.name
        elif layer.name.startswith('relation_embeddings_im'):
            r_im = layer.get_weights()[0];print layer.name
    
    return score_complex_helper( [e_real, e_im, r_real, r_im] , opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval)

def get_complex_kbi_mapping():
    f=open("theo-model/wn18/entity_mapping_wn18.txt");L=f.readlines();f.close()
    map_e_c_o = ddict(int)
    for ele in L:
        ele = ele.strip().split("\t")
        map_e_c_o[int(ele[1])] = int(ele[0])
    f=open("theo-model/wn18/relation_mapping_wn18.txt");L=f.readlines();f.close()
    map_r_c_o = ddict(str)
    for ele in L:
        ele = ele.strip().split("\t")
        map_r_c_o[int(ele[1])] = str(ele[0])

    f=open("/home/cse/phd/csz148211/code/joint_embedding/DATA_REPOSITORY/wn18/original/encoded_data/without-text/relation_id.txt");L=f.readlines();f.close()
    map_r_o_k = ddict(int)
    for ele in L:
        ele = ele.strip().split("\t")
        map_r_o_k[str(ele[0])] = int(ele[1])

    f=open("/home/cse/phd/csz148211/code/joint_embedding/DATA_REPOSITORY/wn18/original/encoded_data/without-text/entity_id.txt");L=f.readlines();f.close()
    map_e_o_k = ddict(int)
    for ele in L:
        ele = ele.strip().split("\t")
        map_e_o_k[int(ele[0])] = int(ele[1])

    map_e_c_k = ddict(int)
    for key in map_e_c_o.keys():
        map_e_c_k[key] = map_e_o_k[map_e_c_o[key]]
    map_r_c_k = ddict(int)
    for key in map_r_c_o.keys():
        map_r_c_k[key] = map_r_o_k[map_r_c_o[key]]

    return map_e_c_k, map_r_c_k

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
    if 0:
        path = "/home/cse/phd/csz148211/code/joint_embedding/code_valid/theo-model/wn18/"#fb15k/"
        print "Using Path", path
        e_real = joblib.load(path+"complex_e1.joblib")
        e_im   = joblib.load(path+"complex_e2.joblib") 
        r_real = joblib.load(path+"complex_r1.joblib")
        r_im   = joblib.load(path+"complex_r2.joblib")

        e_real2 = np.random.randn(e_real.shape[0],e_real.shape[1]); 
        e_im2 = np.random.randn(e_im.shape[0],e_im.shape[1]);
        r_real2 = np.random.randn(r_real.shape[0],r_real.shape[1]);
        r_im2 = np.random.randn(r_im.shape[0],r_im.shape[1]); 

        map_e_c_k, map_r_c_k = get_complex_kbi_mapping() 

        for i in xrange(len(e_real)):
            e_real2[map_e_c_k[int(i)]] = e_real[i]
            e_im2[map_e_c_k[int(i)]] = e_im[i]

        for i in xrange(len(r_real)):
            r_real2[map_r_c_k[i]] = r_real[i]
            r_im2[map_r_c_k[i]] = r_im[i]

    return e_real2, e_im2, r_real2, r_im2

def score_complex_helper(model, opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, complex_eval, oov_flags=None):

    e_real, e_im, r_real, r_im = get_model_weights(model)

    oov_embedding_real = e_real[-1]
    oov_embedding_im = e_im[-1]

    #if os.path.exists("ranks_FB15K_complex.joblib"):
    #    ranks = joblib.load("ranks_FB15K_complex.joblib")

    if 1:#else:
        print strftime("%Y-%m-%d %H:%M:%S", gmtime())
        ranks = complex_eval(e_real, e_im, r_real, r_im, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding_real, oov_embedding_im)
        print strftime("%Y-%m-%d %H:%M:%S", gmtime())
        #for i, comparison_bit in enumerate(ranks[1]):
        #    if (comparison_bit):
        #        ranks[0][i] += oov_e2[i]
        #joblib.dump(ranks, "ranks_FB15K_complex.joblib")
    else:#dec 8
        ranks  = np.array([]); batch_ = 10000
        num_iter = int(testData.shape[0]/batch_)
        print "Using batch size: %d" %batch_
        print "Number of iterations: %d" %num_iter
        start=0;end=start+batch_
        for test_b in xrange(num_iter):
            print test_b, strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ranks_n = complex_eval(e_real, e_im, r_real, r_im, testData[start:end], seen_e2[start:end], oov_flags_1[start:end], oov_flags_2[start:end], oov_embedding_real, oov_embedding_im)
            ranks = np.concatenate([ranks, ranks_n[0]])
            start=end;end+=batch_

        ranks_n = complex_eval(e_real, e_im, r_real, r_im, testData[start:end], seen_e2[start:end], oov_flags_1[start:end], oov_flags_2[start:end], oov_embedding_real, oov_embedding_im)
        ranks = np.concatenate([ranks, ranks_n[0]])
        assert ranks.shape[0] == testData.shape[0]


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


    #file_to_write = "ranks/ranks_complex_"+opts.model+"_"+opts.dataset
    #file_to_write = file_to_write.replace("/", "_")
    #f=open(file_to_write,"w")
    #for ele in ranks[0]:
    #    f.write(str(ele)+"\n")
    #f.close()
 
    '''
    file_to_write = "ranks/raw_"+opts.model+"_"+opts.dataset
    file_to_write = file_to_write.replace("/", "_")
    f=open(file_to_write,"w")
    for ele in ranks[-1]:
        f.write(str(ele)+"\n")
    f.close()
    file_to_write = "ranks/scores_"+opts.model+"_"+opts.dataset
    file_to_write = file_to_write.replace("/", "_")
    f=open(file_to_write,"w");i=0
    for ele in ranks[-2]:
        f.write(str(ranks[-3][i])+"\t"+str(list(ele))+"\n");i+=1
    f.close()'''
    #file_to_write = "ranks/allRank_"+opts.model+"_"+opts.dataset;file_to_write = file_to_write.replace("/", "_")
    #f=open(file_to_write,"w");i=0
    #for ele in ranks[0]:
    #    f.write(str(ele)+"\n");i+=1
    #f.close()
    #if oov_flags is not None:
    #    ranks_oov = []
    #    ranks_others = []
    #    for i in xrange(len(oov_flags)):
    #        if oov_flags[i]:
    #            ranks_oov.append(ranks[0][i])
    #        else:
    #            ranks_others.append(ranks[0][i])

    #    ranks_oov = np.array(ranks_oov)
    #    ranks_others = np.array(ranks_others)
    #    print("NON OOVS: MRR: %5.4f, HITS: %5.4f, HITS@1: %5.4f, MR: %5.4f" %(np.mean(1.0/ranks_others), np.mean(ranks_others <= 10.0), np.mean(ranks_others <= 1.0), np.mean(ranks_others)))
    #    print("OOVS: MRR: %5.4f, HITS: %5.4f, HITS@1: %5.4f, MR: %5.4f" %(np.mean(1.0/ranks_oov), np.mean(ranks_oov <= 10.0),np.mean(ranks_oov <= 1.0), np.mean(ranks_oov)))

    print "FILTERED", np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0), np.mean(ranks[0])
    print "RAW", np.mean(1.0/ranks[-1]), np.mean(ranks[-1] <= 10.0), np.mean(ranks[-1] <= 1.0),np.mean(ranks[-1])
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)
