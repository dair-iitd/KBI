import numpy as np
import random
import theano
import theano.typed_list
import theano.tensor as T
from collections import defaultdict as ddict, Counter
import joblib
import sys, os
import scipy as sp

'''
    For every (e1, r, e2) in the training data check if there are relations r'

    such that (A, r, B) and (B, r' A) exists. Then if (e2, r', e1) exists in training, we can say that (e1, r, e2)

'''
def get_inverse_examples(train_entities, train_relations, testData):

    train_entities_dict = ddict(list)
    train_relation_dict = ddict(list)


    r_scores = ddict(int)


    seen = ddict(list)
    for i, (e1,e2) in enumerate(train_entities):
        r = train_relations[i]
        if (e2,e1) in seen:
            for idx in seen[(e2,e1)]:
                r_inv = train_relations[idx]
                r_scores[(r,r_inv)] += 1
                    

        seen[(e1,e2)].append(i)

    def score(r, r_inv):
        return r_scores[(r,r_inv)] + r_scores[(r_inv, r)]        

    for i, (e1,e2) in enumerate(train_entities):
        train_entities_dict[(e1,e2)].append(i)
        
        curr_relation = train_relations[i]
        train_relation_dict[curr_relation].append((e1,e2))


    symmetric_r = set()
    inv_relation_arr = []
   

    for i, test_fact in enumerate(testData):
        e1, r, e2 = test_fact


        best_inv_score = -1
        best_inv_relation = None
        symmetric = False

        if (e2, e1) in train_entities_dict:
            for idx in train_entities_dict[(e2,e1)]:
                candidate_relation = train_relations[idx]
                if (candidate_relation == r):
                    symmetric = True
                else:    
                    score_curr = score(r, candidate_relation)
                    if (score_curr > best_inv_score):
                        best_inv_relation = candidate_relation 
                        best_inv_score = score_curr

        if symmetric: symmetric_r.add(r)

        inv_relation_arr.append((best_inv_score, best_inv_relation))

    return symmetric_r, inv_relation_arr


'''
    perform some error analysis given the ranks and the testData

'''

def get_id_to_name(name_idx, id_idx, f_name):
    f = open(f_name, "r")
    all_f = f.readlines(); f.close()

    IdToName = {}

    for i, line in enumerate(all_f):
        lineSep = line.split("\t")
        id = int(lineSep[id_idx])

        IdToName[id] = lineSep[name_idx]
    
    return IdToName


def print_preds(preds, testData):
    entityIdToName = get_id_to_name(1, 2, "lib/ent_wiki_mapping.txt")
    relationIdToName = get_id_to_name(0, 1, "lib/relation_id.txt")


    for i in xrange(len(testData)):
        print("************************")
        e1, r, e2 = testData[i] 
        e1 = entityIdToName[e1]
        e2 = entityIdToName[e2]
        r = relationIdToName[r]
        print("%s %s" %(e1,r))
        e2s_predicted, _  = preds[i]
        for e2_predicted in e2s_predicted:
            e2_predicted = entityIdToName[e2_predicted]
            print("%s - %s" %(e2, e2_predicted))

 
def analyse(ranks, preds,label, test_scores,  trainData, testData, symmetric_r, inv_relation_arr, inv_scores):

    entityIdToName = get_id_to_name(1, 2, "lib/ent_wiki_mapping.txt")
    relationIdToName = get_id_to_name(0, 1, "lib/relation_id.txt")

    train_entities, train_relations = trainData

    entity_freq = ddict(int)
    relation_freq = ddict(int)

    for i, (e1,e2) in enumerate(train_entities): 
        entity_freq[e1] += 1
        entity_freq[e2] += 1
        r = train_relations[i]
        relation_freq[r] += 1

    

    def pretty_print(sranks, preds_best):
        itr = 0
        symmetric=0
        inverses=0
        avg_e1_freq=0
        avg_e2_freq=0
        avg_r_freq=0
        avg_r_inv=0
        avg_min_r_r_inv=0

        for curr_rank, i in sranks:
            e1,r,e2 = testData[i] 
            e1_name = entityIdToName[e1] 
            e2_name = entityIdToName[e2] 
            r_name = relationIdToName[r]     
            freq_e1 = entity_freq[e1]
            freq_e2 = entity_freq[e2]           
            freq_r = relation_freq[r]

            avg_e1_freq += freq_e1
            avg_e2_freq += freq_e2
            avg_r_freq += freq_r

            print("*"*50)
            pred_arr, score = preds_best[itr]
            for pred in pred_arr:        
                best_pred_name = entityIdToName[pred]
                print "PREDICTION: %s, score: %5.4f " %(best_pred_name, score),
    

            print

            testpoint_score = test_scores[i]
            if r in symmetric_r:
                symmetric += 1     
                print("SYMMETRIC REL: (%s, %s, %s) | e1freq=%d, rfreq=%d ,e2freq=%d | rank = %d, score = %5.4f " %(e1_name ,r_name ,e2_name, freq_e1, freq_r, freq_e2, curr_rank, testpoint_score))
            else:
                score, inv_r = inv_relation_arr[i]
                if inv_r is not None:
                    inv_score = inv_scores[i]
                    freq_inv = relation_freq[inv_r]
                    freq_r_r_min = min(freq_r, freq_inv)
                    avg_min_r_r_inv += freq_r_r_min
                    avg_r_inv += freq_inv
                    inverses+=1
                    inv_r_name = relationIdToName[inv_r]
                    print("INV REL: %s - %d: freq = %d, score = %5.4f | (%s, %s, %s) | e1freq=%d, rfreq=%d, e2freq=%d  | rank = %d, score = %5.4f " %(inv_r_name, score, freq_inv, inv_score, e1_name ,r_name ,e2_name, freq_e1, freq_r, freq_e2, curr_rank, testpoint_score))
                else:
                    print("(%s, %s, %s) | e1freq=%d, rfreq=%d, e2freq=%d | rank = %d, score = %5.4f" %(e1_name ,r_name ,e2_name, freq_e1, freq_r, freq_e2, curr_rank, testpoint_score))
                    
            itr += 1              

        avg_e1_freq /= 100.0
        avg_e2_freq /= 100.0
        avg_r_freq /= 100.0
        avg_r_freq /= inverses
        avg_min_r_r_inv /= inverses


        print("Symmetric relations: %d, Inverse relations: %d, AVERAGE e1 freq: %5.4f, AVERAGE r freq: %5.4f, AVERAGE e2 freq: %5.4f, AVERAGE r inv: %5.4f, AVERAGE min(r,r-inv): %5.4f" %(symmetric, inverses, avg_e1_freq, avg_r_freq, avg_e2_freq, avg_r_inv, avg_min_r_r_inv))


    print("--------------------%s--------------------" %label)
    pretty_print(ranks, preds)

         



def create_oov_vector(entity_weights, train_entities):
    entities = [e for (e1,e2) in train_entities for e in (e1,e2)] 

    entity_counter = Counter(entities)
    oov_embedding = np.zeros(entity_weights.shape[1])
    cnt=0.0
    for item in entity_counter:
        if (entity_counter[item] == 1):
            cnt+=1.0
            oov_embedding += entity_weights[item]

    oov_embedding /= cnt
    return oov_embedding 

def E_oovEval():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_flags = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()

    def E_fn(testPoint, oov_flag, i, entities, relations, seen_e2):
        scores = T.dot(entities, relations[testPoint[1]])
        score_testPoint = T.switch(oov_flag,T.dot(oov_embedding, relations[testPoint[1]]), T.dot(entities[testPoint[2]], relations[testPoint[1]]))
        rank = 1 + T.sum(scores > score_testPoint)-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(oov_embedding, relations[testPoint[1]])
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison

    ranks, ignore = theano.scan(E_fn, non_sequences = [entities,relations, seen_e2], sequences = [testData, oov_flags, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations,testData, seen_e2, oov_flags, oov_embedding], ranks,allow_input_downcast=True)
    return f
    

def scoreE_OOV(model, train_entities, testData, oov_flags, seen_e2, oov_e2, E_oovEval):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights = layer.get_weights()[0]
    
    vect_dim = entity_weights.shape[1]
    oov_embedding = create_oov_vector(entity_weights, train_entities)
    ranks = E_oovEval(entity_weights, relation_weights, testData, seen_e2, oov_flags, oov_embedding)

    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags[i]):
            ranks[0][i] += oov_e2[i]
         
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)    


def forward_pass_func(layers, inp):
    for layer in layers[:-1]:
        W, b = layer
        inp = T.tanh(T.dot(inp, W) + b)

    W, b = layers[-1]
    return T.nnet.sigmoid(T.dot(inp, W) + b)

def concat_oovEval():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    
    nnet_layer_1_weights = T.dmatrix()
    nnet_layer_1_bias    = T.dvector() 
    nnet_layer_2_weights = T.dmatrix()
    nnet_layer_2_bias    = T.dvector() 
    nnet_layer_3_weights = T.dmatrix()
    nnet_layer_3_bias    = T.dvector()
    
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()

    layers = [(nnet_layer_1_weights, nnet_layer_1_bias),(nnet_layer_2_weights, nnet_layer_2_bias),(nnet_layer_3_weights, nnet_layer_3_bias)]

    def concat_fn(testPoint, oov_flag1, oov_flag2, i, entities, relations, nnet_layer_1_weights, nnet_layer_1_bias, nnet_layer_2_weights, nnet_layer_2_bias, nnet_layer_3_weights, nnet_layer_3_bias, seen_e2):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e2_fact_embedding = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])
        
        e1_r              = T.concatenate([e1_fact_embedding.dimshuffle('x',0), relations[testPoint[1]].dimshuffle('x',0)], axis=-1)
        e1_r_tile         = T.tile(e1_r, reps=(entities.shape[0], 1))#.dimshuffle(0,'x').dimshuffle(1,0), reps=(1,entities.shape[0]))
        e1_r_all          = T.concatenate([e1_r_tile, entities], axis=-1)
        e1_r_e2           = T.concatenate([e1_r, e2_fact_embedding.dimshuffle('x',0)], axis=-1) 
        scores            = forward_pass_func(layers, e1_r_all)
        score_testPoint   = forward_pass_func(layers, e1_r_e2)[0]
        
        rank = 1 + T.sum(scores > score_testPoint) - T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison

    ranks, ignore = theano.scan(concat_fn, non_sequences = [entities,relations, nnet_layer_1_weights, nnet_layer_1_bias, nnet_layer_2_weights, nnet_layer_2_bias, nnet_layer_3_weights, nnet_layer_3_bias, seen_e2], sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations, nnet_layer_1_weights, nnet_layer_1_bias, nnet_layer_2_weights, nnet_layer_2_bias, nnet_layer_3_weights, nnet_layer_3_bias, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding], ranks,allow_input_downcast=True)
    return f


def L2_norm(data,axis):
    return T.sqrt(T.sum(T.sqr(data),axis))

def L1_norm(data,axis):
    return T.sum(data,axis)

def TransE_oovEval():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    
    def TransE_fn(testPoint, oov_flag1, oov_flag2, i, entities, relations, seen_e2):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e2_fact_embedding = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])
        e1_minus_e2_all   = e1_fact_embedding - entities
        scores     = -1.0 * T.sqrt(T.sum(T.sqr(e1_minus_e2_all + relations[testPoint[1]]),1))
        score_testPoint   = -1.0 * T.sqrt(T.sum(T.sqr(relations[testPoint[1]] + e1_fact_embedding - e2_fact_embedding)))
        
        rank = 1 + T.sum(scores > score_testPoint)-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison        
        
    ranks, ignore = theano.scan(TransE_fn, non_sequences = [entities,relations, seen_e2], sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations,testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding], ranks,allow_input_downcast=True)
    return f

def scoreTransE_OOV(opts, model, testData, train_entities, oov_flags_1, oov_flags_2, seen_e2, oov_e2, TransE_fn):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
    
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1] 
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)

    #ranks = TransE_fn(entity_weights, relation_weights, testData, triples_known)
    ranks = TransE_fn(entity_weights, relation_weights, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding)
    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags_2[i]):
            ranks[0][i] += oov_e2[i]
         
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)  


def distMult_run():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()

    def distMult_fn(testPoint, entities, relations):
        e1_embedding = entities[testPoint[0]]
        e2_embedding = entities[testPoint[2]]
        return T.dot(e1_embedding*e2_embedding, relations[testPoint[1]])

    scoredm, ignore = theano.scan(distMult_fn, non_sequences = [entities,relations], sequences = [testData])
    f = theano.function([entities,relations,testData], scoredm,allow_input_downcast=True)

    return f

def distMult_get_scores():
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

def distMult_oovEval():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    oov_e2 = T.ivector()

    def distMult_fn(testPoint, oov_flag1, oov_flag2, i, entities, relations, seen_e2, oov_e2):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e2_fact_embedding = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])
        e1_cross_e2_all  = e1_fact_embedding*entities
        scores   = T.dot(e1_cross_e2_all, relations[testPoint[1]])
        score_testPoint   = T.dot(relations[testPoint[1]], e1_fact_embedding*e2_fact_embedding) 
        rank = 1 + T.sum(scores > score_testPoint)-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_oov > score_testPoint
        rank += same/2.0 # add a penalty if some scores are the same

        rank = T.switch(oov_comparison, rank + oov_e2[i], rank) # add a penalty if not OOV and lost to all
        rank = T.switch(oov_flag, rank + oov_e2[i]/2.0, rank) # add a penalty if OOV

        return rank, oov_comparison, score_testPoint

    ranks, ignore = theano.scan(distMult_fn, non_sequences = [entities,relations, seen_e2, oov_e2], sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations,testData, seen_e2, oov_e2, oov_flags_1, oov_flags_2, oov_embedding], ranks,allow_input_downcast=True)
    return f


def probScoreDM_OOV(model, opts, kb_entities, kb_relations, neg_samples_kb, oov_flags = None):
    train_entities, train_relations = trainData
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1]
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations

    entity_vectors = entity_weights[kb_entities]
    entity_negative_vectors = entity_weights[neg_samples_kb]
    relation_vectors = relation_weights[kb_relations]
        

def scoreDM_OOV(model, opts, testData, trainData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, distMult_oovEval, distMult_get_scores, distMult_run, oov_flags = None):
    print("evaluating.")
    train_entities, train_relations = trainData   
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
    
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1] 
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)
   
    ranks = distMult_oovEval(entity_weights, relation_weights, testData, seen_e2, oov_e2, oov_flags_1, oov_flags_2, oov_embedding)


    mrr  = np.mean(1.0/ranks[0])
    hits = np.mean(ranks[0] <= 10.0)    



    return mrr, hits    

def scoreConcat_OOV(model, opts, testData, train_entities, oov_flags_1, oov_flags_2, seen_e2, oov_e2, concat_oovEval, oov_flags = None):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'dense_1':
            nnet_layer_1_weights = layer.get_weights()[0]
            nnet_layer_1_bias = layer.get_weights()[1]
        elif layer.name == 'dense_2':
            nnet_layer_2_weights = layer.get_weights()[0]
            nnet_layer_2_bias = layer.get_weights()[1]
        elif layer.name == 'dense_3':
            nnet_layer_3_weights = layer.get_weights()[0]
            nnet_layer_3_bias = layer.get_weights()[1]
        
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1] 
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)
    
    ranks = concat_oovEval(entity_weights, relation_weights, nnet_layer_1_weights, nnet_layer_1_bias, nnet_layer_2_weights, nnet_layer_2_bias, nnet_layer_3_weights, nnet_layer_3_bias, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding)

    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags_2[i]):
            ranks[0][i] += oov_e2[i]
        
    if oov_flags is not None:
        ranks_oov = []
        ranks_others = []
        for i in xrange(len(oov_flags)):
            if oov_flags[i]:
                ranks_oov.append(ranks[0][i])
            else:
                ranks_others.append(ranks[0][i])
 
        ranks_oov = np.array(ranks_oov)
        ranks_others = np.array(ranks_others)
        print("NON OOVS: MRR: %5.4f, HITS: %5.4f" %(np.mean(1.0/ranks_others), np.mean(ranks_others <= 10.0)))
        print("OOVS: MRR: %5.4f, HITS: %5.4f" %(np.mean(1.0/ranks_oov), np.mean(ranks_oov <= 10.0)))
    
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)    




