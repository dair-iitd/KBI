import numpy as np
import random
import theano
import theano.typed_list
import theano.tensor as T
from collections import defaultdict as ddict, Counter
import joblib
import sys, os
import scipy as sp
from time import gmtime,strftime

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


def typedDM_oovEval():
    entities = T.dmatrix()
    relations = T.dmatrix()
    entity_types = T.dmatrix()
    relation_head_types = T.dmatrix()
    relation_tail_types = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_type_embedding = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    oov_e2 = T.ivector()
    def tpf(testPoint, oov_flag1, oov_flag2, i, entities, relations, entitiy_types, relation_head_types, relation_tail_types, seen_e2, oov_e2):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e1_type_embedding = T.switch(oov_flag1, oov_type_embedding, entity_types[testPoint[0]])
        e2_fact_embedding = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])
        e2_type_embedding = T.switch(oov_flag2, oov_type_embedding, entity_types[testPoint[2]])
        e1_cross_all_e2 = e1_fact_embedding*entities
        dm_scores = T.dot(e1_cross_all_e2, relations[testPoint[1]])
        dm_scores = 1/(1+T.exp(-dm_scores))
        #dm_scores = 100+dm_scores
        dm_score_testPoint = T.dot(relations[testPoint[1]], e1_fact_embedding*e2_fact_embedding)
        dm_score_testPoint = 1/(1+T.exp(-dm_score_testPoint))
        head_type_factor = T.dot(relation_head_types[testPoint[1]], e1_type_embedding)
        head_type_factor = 1/(1+T.exp(-head_type_factor))
        #type_score_testPoint = head_type_factor*(1/(1+T.exp(-T.dot(relation_tail_types[testPoint[1]], e2_type_embedding))))
        #type_scores = head_type_factor*(1/(1+T.exp(-T.dot(entity_types, relation_tail_types[testPoint[1]]))))
        type_score_testPoint = (1/(1+T.exp(-T.dot(relation_tail_types[testPoint[1]], e2_type_embedding))))
        type_scores = (1/(1+T.exp(-T.dot(entity_types, relation_tail_types[testPoint[1]]))))
        #type_score_testPoint = type_scores[testPoint[2]]
        scores = dm_scores*type_scores
        score_testPoint=dm_score_testPoint*type_score_testPoint
        rank1 = 1+T.sum(scores > score_testPoint)
        rank = rank1-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]], score_testPoint))
        rank += same/2.0
        return rank, scores, rank1#, testPoint[0], e1_fact_embedding,testPoint[2], e2_fact_embedding
    rank, ignore = theano.scan(tpf, non_sequences=[entities, relations, entity_types, relation_head_types, relation_tail_types, seen_e2, oov_e2], sequences=[testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities, relations, entity_types, relation_head_types, relation_tail_types, testData, seen_e2, oov_e2, oov_flags_1, oov_flags_2, oov_embedding, oov_type_embedding], rank, allow_input_downcast=True, on_unused_input='warn')
    return f




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
        rank1 = 1 + T.sum(scores > score_testPoint)
        rank = rank1 -T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(oov_embedding, relations[testPoint[1]])
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison,rank1

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
    print "FILTERED:", np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0),np.mean(ranks[0])
    print "RAW:", np.mean(1.0/ranks[2]), np.mean(ranks[2] <= 10.0), np.mean(ranks[2] <= 1.0),np.mean(ranks[2])
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)

def scoreTDM_OOV(model, opts, testData, trainData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, typedDM_oovEval, oov_flags = None):
    #print("\nevaluating")
    train_entities, train_relations = trainData
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'entity_type_embeddings':
            entity_type_embeddings = layer.get_weights()[0]
        elif layer.name == 'relation_head_type_embeddings':
            relation_head_type_embeddings = layer.get_weights()[0]
        elif layer.name == 'relation_tail_type_embeddings':
            relation_tail_type_embeddings = layer.get_weights()[0]
    vect_dim = entity_weights.shape[1]
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1]
        oov_type_embedding = entity_type_embeddings[-1]
    elif opts.oov_average:
        oov_embedding = create_oov_vector(entity_weights, train_entities)
        oov_type_embedding = create_oov_vector(entity_type_embeddings, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)
        oov_type_embedding = np.random.rand(vect_dim)
    ranks = typedDM_oovEval(entity_weights, relation_weights, entity_type_embeddings, relation_head_type_embeddings, relation_tail_type_embeddings, testData, seen_e2, oov_e2, oov_flags_1, oov_flags_2, oov_embedding, oov_type_embedding)
    mrr  = np.mean(1.0/ranks[0])
    hits = np.mean(ranks[0] <= 10.0)

    print "FILTERED:", np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0),np.mean(ranks[0])
    print "RAW:", np.mean(1.0/ranks[2]), np.mean(ranks[2] <= 10.0), np.mean(ranks[2] <= 1.0),np.mean(ranks[2])

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


    print("dddddddddddddddddddddddddddddd", np.mean(ranks[0] <= 1.0))
    print("special")
    scores = ranks[1]
    #np.savetxt("scores.csv", scores, delimiter=",")
    #all_ranks = sp.stats.rankdata(scores)
    top_10 = scores.argsort()[:, -10:]
    #np.savetxt("top_10.csv", top_10, fmt="%8.0f", delimiter=",")
    #np.savetxt("ranks.csv", ranks[0], fmt="%8.0f", delimiter=",")
    print("done")

    return mrr, hits

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

        rank1 = 1 + T.sum(scores > score_testPoint)
        rank = rank1-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison, rank1

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
    print "FILTERED ::", np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0), np.mean(ranks[0])
    print "RAW ::", np.mean(1.0/ranks[2]), np.mean(ranks[2] <= 10.0), np.mean(ranks[2] <= 1.0), np.mean(ranks[2])
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

    def distMult_fn(testPoint, oov_flag1, oov_flag2, i, entities, relations, seen_e2,oov_e2):
        e1_fact_embedding = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e2_fact_embedding = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])
        e1_cross_e2_all  = e1_fact_embedding*entities
        scores   = T.dot(e1_cross_e2_all, relations[testPoint[1]])
        score_testPoint   = T.dot(relations[testPoint[1]], e1_fact_embedding*e2_fact_embedding)
        rank1 = 1 + T.sum(scores > score_testPoint)
        rank = rank1-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        rank = T.switch(oov_comparison, rank, rank + oov_e2[i])
        rank = T.switch(oov_flag2, rank + oov_e2[i]/2.0, rank)
        return rank, oov_comparison, score_testPoint,rank1, scores

    ranks, ignore = theano.scan(distMult_fn, non_sequences = [entities,relations, seen_e2,oov_e2], sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations,testData, seen_e2,oov_e2, oov_flags_1, oov_flags_2, oov_embedding], ranks,allow_input_downcast=True)
    return f

def attention_distMult_oovEval():
    entities  = T.dmatrix()
    attention = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    oov_embedding = T.dvector()
    oov_embedding_attention = T.dvector()
    oov_flags_1 = T.ivector() # 1 if e1 is oov else 0
    oov_flags_2 = T.ivector() # 1 if e2 is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()

    def distMult_fn(testPoint, oov_flag1, oov_flag2, i, entities, attention, relations, seen_e2):
        e1_fact_matrix = T.switch(oov_flag1, oov_embedding, entities[testPoint[0]])
        e2_fact_matrix = T.switch(oov_flag2, oov_embedding, entities[testPoint[2]])

        e1_attention_fact_matrix = T.switch(oov_flag1, oov_embedding_attention, attention[testPoint[0]])
        e2_attention_fact_matrix = T.switch(oov_flag1, oov_embedding_attention, attention[testPoint[2]])

        attention_e1_fact_matrix = e1_fact_matrix * e1_attention_fact_matrix
        attention_e2_fact_matrix = e2_fact_matrix * e2_attention_fact_matrix

        attention_e1_fact_matrix_r = attention_e1_fact_matrix.reshape((3,2))#topic x dim
        attention_e1_fact_matrix_r = attention_e1_fact_matrix.reshape((3,2))

        e1_cross_e2_all  = e1_fact_embedding*entities
        scores   = T.dot(e1_cross_e2_all, relations[testPoint[1]])
        score_testPoint   = T.dot(relations[testPoint[1]], e1_fact_embedding*e2_fact_embedding)
        rank = 1 + T.sum(scores > score_testPoint)-T.sum(scores[seen_e2[i]] > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint)) - T.sum(T.eq(scores[seen_e2[i]],score_testPoint))
        score_oov = T.dot(relations[testPoint[1]], e1_fact_embedding*oov_embedding)
        oov_comparison = score_testPoint > score_oov
        rank += same/2.0
        return rank, oov_comparison, score_testPoint

    ranks, ignore = theano.scan(distMult_fn, non_sequences = [entities,attention,relations, seen_e2], sequences = [testData, oov_flags_1, oov_flags_2, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,attention,relations,testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding, oov_embedding_attention], ranks,allow_input_downcast=True)
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
        #print "###", layer.name
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]; #print "I am here!!", layer.name
            #print layer.get_weights()[0][11292]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]; #print "I am here!!", layer.name

    #print "Test Data", testData[0]
    #print "\nEntity: e1: \n ", entity_weights[testData[0][0]]
    #print "\nEntity: e2: \n ", entity_weights[testData[0][2]]
    #print "\nRelations: r: \n ",relation_weights[testData[0][1]]

    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1]
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)

    if 0:#os.path.exists("ranks_FB15K_DM.joblib"):
        ranks = joblib.load("ranks_FB15K_DM.joblib")

    if 0:#else:#dec 8
        ranks  = np.array([]); batch_ = 6000
        num_iter = int(testData.shape[0]/batch_)
        print "Using batch size: %d" %batch_
        print "Number of iterations: %d" %num_iter
        start=0;end=start+batch_
        for test_b in xrange(num_iter):
            print test_b, strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ranks_n = distMult_oovEval(entity_weights, relation_weights, testData[start:end], seen_e2[start:end], oov_e2[start:end], oov_flags_1[start:end], oov_flags_2[start:end], oov_embedding)
            ranks = np.concatenate([ranks, ranks_n[0]])
            start=end;end+=batch_
        ranks_n = distMult_oovEval(entity_weights, relation_weights, testData[start:end], seen_e2[start:end], oov_e2[start:end], oov_flags_1[start:end], oov_flags_2[start:end], oov_embedding)
        ranks = np.concatenate([ranks, ranks_n[0]])
        assert ranks.shape[0] == testData.shape[0]

    #ranks = [ranks]#dec 8

    ranks = distMult_oovEval(entity_weights, relation_weights, testData, seen_e2, oov_e2, oov_flags_1, oov_flags_2, oov_embedding)

    '''
    print "Model:: EW :: RW :: ", entity_weights, relation_weights,

    print "RANKS :: ", ranks, ranks[0].shape
    print "RANKS :: ", ranks[0][:10]
    print "OOV_C :: ", ranks[1][:10]
    print "Scores :: ", ranks[2][:10], ranks[2].shape
    print entity_weights.shape
    print relation_weights.shape

    print testData[0]

    same=0;small=0

    print "all_scores"
    test_point_score = np.dot(entity_weights[testData[0][2]] * entity_weights[testData[0][0]], relation_weights[testData[0][1]])
    for i in xrange(entity_weights.shape[0]):
        s=np.dot(entity_weights[i] * entity_weights[testData[0][0]], relation_weights[testData[0][1]])
        print i, s
        if s == test_point_score and i != testData[0][2]:
            same+=1
        elif s>test_point_score:
            small+=1

    print small, same
    '''
       #joblib.dump(ranks, "ranks_FB15K_DM.joblib")
    '''
    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags_2[i]):
            ranks[0][i] += oov_e2[i]

    def get_best(score_vector, seen_e2_curr, e2):
        best_pred = None
        best_score = -1<<20

        for i, score in enumerate(score_vector):
            if i not in seen_e2_curr or i == e2:
                if (score > best_score):
                    best_score = score; best_pred = i


        all_best = []

        for i, score in enumerate(score_vector):
            if score == best_score: all_best.append(i)

        return all_best, best_score

    def get_preds(rank_id_pairs):
        test_curr = []; seen_e2_curr = []; oov_flags_1_curr = []

        for (rank, i) in rank_id_pairs:
            test_curr.append(testData[i]);
            seen_e2_curr.append(seen_e2[i]);
            oov_flags_1_curr.append(oov_flags_1[i]);

        score_matrix = distMult_get_scores(entity_weights, relation_weights, test_curr, oov_flags_1_curr, oov_embedding)
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


        score_distMult = distMult_run(entity_weights, relation_weights, test_curr)
        return score_distMult


    symmetric_r, inv_relation_arr = get_inverse_examples(train_entities, train_relations, testData)

    entityIdToName = get_id_to_name(1, 2, "lib/ent_wiki_mapping.txt")
    relationIdToName = get_id_to_name(0, 1, "lib/relation_id.txt")
    '''
    mrr  = np.mean(1.0/ranks[0])
    hits = np.mean(ranks[0] <= 10.0)
    hits1 = np.mean(ranks[0] <= 1.0)
    mr = np.mean(ranks[0])
    print("mrr: %5.4f, hits@10: %5.4f, hits@1: %5.4f, MR: %5.4f" %(mrr, hits, hits1, mr))
    '''
    entity_freq = ddict(int)
    e1_mrr_dict = ddict(list)

    for i, (e1,e2) in enumerate(train_entities):
        entity_freq[e1] += 1
        entity_freq[e2] += 1

    high_e1=[];low_e1=[]
    high_r=[];low_r=[]
    print ranks[0].shape, testData.shape
    for i, test_point in enumerate(testData):
        e1, r1, e2 = test_point
        if ranks[0][i] < 3:
            high_e1.append(entity_freq[e1])
        else:
            low_e1.append(entity_freq[e1])

        if entity_freq[e1] < 200:
            high_r.append(ranks[0][i])
        else:
            low_r.append(ranks[0][i])

    high_e1, low_e1 = (np.array(high_e1), np.array(low_e1))
    high_r, low_r = (np.array(high_r), np.array(low_r))
    print "e1 fre for rank <3 and >3", np.mean(high_e1), np.mean(low_e1)
    print "fact rank for e1Freq <200 and >200", np.mean(high_r), np.mean(low_r)'''
    '''
    #@Shikhar: Some bug here
    for i, test_point in enumerate(testData):
        e1, r, e2 = test_point
        e1_mrr_dict[e1].append(1.0/ranks[0][i])


    f = open("mrr_freq.txt", "w")
    for ent in e1_mrr_dict:

        avg_mrr = sum(e1_mrr_dict[ent])/float(len(e1_mrr_dict[ent]))
        f.write("%s: %d, %5.4f\n" %(entityIdToName[ent], entity_freq[ent], avg_mrr))

    f.close()

    f = open("inverse_relations.txt", "w")
    for i, test_point in enumerate(testData):
       e1, r1, e2 = test_point
       score, r2 = inv_relation_arr[i]
       if r2 is not None:
        cosine_dist = sp.spatial.distance.cosine(relation_weights[r1], relation_weights[r2])
        f.write("%s \t %s : %d: %5.4f\n" %(relationIdToName[r1], relationIdToName[r2], score, cosine_dist))

    f.close()


    cosine_dist = sp.spatial.distance.cosine(entity_weights[6731], entity_weights[3474])
    print(cosine_dist)
    '''
    #sorted_ranks = sorted(zip(ranks[0], range(len(testData))))

    #analysis_points = [(rank, i) for (rank, i) in sorted_ranks]
    #preds = get_preds(analysis_points)
    #for pred in preds:
    #    curr_pred, _ = pred
    #    if (len(curr_pred) > 1): print("Here")

    #print_preds(preds, testData)

#    confusion_britain_and_us = 0
#    for i, pred in enumerate(preds):
#        e2 = testData[analysis_points[i][1]][2]
#        if ((pred[0] == 6731 and e2 == 3474) or (pred[0] == 3474 and e2 == 6731)):
#            confusion_britain_and_us += 1
#
#    confusion_britain_and_us /= float(len(analysis_points))
#    print("confusion: %5.4f" %(confusion_britain_and_us))
#
#    symmetric_r, inv_relation_arr = get_inverse_examples(train_entities, train_relations, testData)
#    inv_scores = get_inverse_scores(inv_relation_arr)
#    analyse(analysis_points,preds, "Predictions", ranks[2], trainData, testData, symmetric_r, inv_relation_arr, inv_scores)
    print ranks[0].shape,len(oov_flags)
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

    print "FILTERED", np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0), np.mean(ranks[0])
    print "RAW", np.mean(1.0/ranks[3]), np.mean(ranks[3] <= 10.0), np.mean(ranks[3] <= 1.0), np.mean(ranks[3])
    scores = ranks[4]
    print("dddddddddddddddddddddddddddddd", np.mean(ranks[0] <= 1.0))
    print("special")
    #np.savetxt("scores.csv", scores, delimiter=",")
    #all_ranks = sp.stats.rankdata(scores)
    top_10 = scores.argsort()[:, -10:]
    #np.savetxt("top_10_dm.csv", top_10, fmt="%8.0f", delimiter=",")
    #np.savetxt("ranks_dm.csv", ranks[0], fmt="%8.0f", delimiter=",")
    print("done")
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)


def scoreAttentionDM_OOV(model, opts, testData, trainData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, distMult_oovEval, distMult_get_scores, distMult_run, oov_flags = None):
    print("evaluating.")
    train_entities, train_relations = trainData
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'entities_attention':
            entity_attention_weights = layer.get_weights()[0]

    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1]
        oov_embedding_attention = entity_attention_weights[-1]
    elif opts.oov_average:
        oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)

    ranks = attention_distMult_oovEval(entity_weights, entity_attention_weights, relation_weights, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding, oov_embedding_attention)


    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags_2[i]):
            ranks[0][i] += oov_e2[i]
    print np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0)
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)

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
    print np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0), np.mean(ranks[0] <= 1.0), np.mean(ranks[0])
    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)
