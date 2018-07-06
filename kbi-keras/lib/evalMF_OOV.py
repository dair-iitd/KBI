import numpy as np
import random
import sys
import theano 
import theano.tensor as T

def ddict_set():
    return ddict(set)

def get_relations(testData, oov_flag=[]):
    #relation_map = ddict(set)
    #map(lambda test_fact, oov: relation_map[test_fact[0]].add((test_fact[1],oov)), testData, oov_flag)
    relation_map = ddict(ddict_set)
    map(lambda test_fact, oov: relation_map[test_fact[0]][test_fact[1]].add(oov), testData, oov_flag)

    return relation_map
    

def MF_oovEval_naive():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    entity_oov_embedding = T.dvector()
    oov_flags = T.ivector() # 1 if (e1,e2) is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    oovs = T.ivector()

    # testPoint is relationId, entityPairId
    def MF_fn(testPoint, oov_flag, i, entities, relations, entity_oov_embedding, seen_e2, oovs):
        scores = T.dot(entities[seen_e2[i]], relations[testPoint[0]])
        score_oov = T.dot(entity_oov_embedding,relations[testPoint[0]]) 
        score_testPoint = T.switch(oov_flag, score_oov, T.dot(entities[testPoint[1]], relations[testPoint[0]]))
        rank = 1 + T.sum(scores > score_testPoint)
        same = T.sum(T.eq(scores, score_testPoint))
        rank += same/2.0

        oov_comparison = score_oov > score_testPoint 
        rank = T.switch(oov_comparison, rank + oovs[i], rank)
        rank = T.switch(oov_flag, rank + oovs[i]/2.0, rank)
        return rank

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities,relations, entity_oov_embedding, seen_e2, oovs], sequences = [testData, oov_flags, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations, entity_oov_embedding, testData, seen_e2, oovs, oov_flags], ranks,allow_input_downcast=True)
    return f


'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_counts[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def scoreMF_OOV_naive(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]

    vect_dim = entity_weights.shape[1]

    if opts.oov_train:
        print "using trained oov embedding"
        oov_embedding = entity_weights[-1]
    elif opts.oov_average:
        print "using average OOV embedding"
        oov_embedding = np.zeros(vect_dim)
        num_singletons = 0
        for i in xrange(opts.num_entities):#entity_weights.shape[0]-1):
            if counter[i] == 1:
                oov_embedding += entity_weights[i]
                num_singletons += 1.0

        oov_embedding /= num_singletons 

    else:
        print "using random oov embedding"
        vect_dim = entity_weights.shape[1]
        oov_embedding = np.random.randn(vect_dim)


    ranks = MF_oovEval(entity_weights, relation_weights, oov_embedding, testData, seen_e2, oov_counts, oov_flags)         
    oov_ranks = np.array([rank for i, rank in enumerate(ranks) if oov_flags[i]])
    seen_ranks = np.array([rank for i, rank in enumerate(ranks) if not oov_flags[i]])
 
    print_score(oov_ranks)
    print_score(seen_ranks)   

    if 0:
        def write_rank(data, file_name):
            f=open(file_name, "w")
            for i in data:
                f.write(str(i)+"\n")
            f.close()
        file_name = "ranks/%s_%s_ranks.txt" %(opts.model_path.split("_")[1],opts.model)
        write_rank(ranks, file_name.lower())

    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)

def print_score(ranks):
    mrr = np.mean(1.0/ranks)
    hits = np.mean(ranks <= 10.0)
    print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))

def jointMF_oovEval_naive():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    entity_oov_embedding = T.dvector()
    oov_flags = T.ivector() # 1 if (e1,e2) is oov else 0
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    oovs = T.ivector()

    # testPoint is relationId, entityPairId
    def MF_fn(testPoint, oov_flag, i, entities, relations, entity_oov_embedding, seen_e2, oovs):
        scores = T.tanh(T.dot(entities[seen_e2[i]], relations[testPoint[0]]))
        score_oov = T.tanh(T.dot(entity_oov_embedding,relations[testPoint[0]])) 
        score_testPoint = T.switch(oov_flag, score_oov, T.dot(entities[testPoint[1]], relations[testPoint[0]]))
        rank = 1 + T.sum(scores > score_testPoint)
        oov_comparison = score_oov > score_testPoint 
        rank = T.switch(oov_comparison, rank + oovs[i], rank)
        rank = T.switch(oov_flag, rank + oovs[i]/2.0, rank)
        return rank

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities,relations, entity_oov_embedding, seen_e2, oovs], sequences = [testData, oov_flags, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations, entity_oov_embedding, testData, seen_e2, oovs, oov_flags], ranks,allow_input_downcast=True)
    return f


def scoreMF_joint_OOV_naive(model, opts, testData, oov_flags, seen_e2, oov_counts, joint_oovEval):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings_MF':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]

    vect_dim = entity_weights.shape[1]

    if not opts.oov_eval:
        print "using incorrect evaluation"

    if opts.oov_train:
        print "using trained oov embedding"
        oov_embedding = entity_weights[-1]
    elif opts.oov_average:
        print "using average OOV embedding"
        oov_embedding = np.zeros(vect_dim)
        for i in xrange(opts.num_entity_pairs):#entity_weights.shape[0]-1):
            oov_embedding += entity_weights[i]

        oov_embedding /= (entity_weights.shape[0])

    else:
        print "using random oov embedding"
        vect_dim = entity_weights.shape[1]
        oov_embedding = np.random.randn(vect_dim)


    ranks = joint_oovEval(entity_weights, relation_weights, oov_embedding, testData, seen_e2, oov_counts, oov_flags)         
    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)


def MF_eval():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()

    # testPoint is relationId, entityPairId
    def MF_fn(testPoint, i, entities, relations, seen_e2):
        scores = T.dot(entities[seen_e2[i]], relations[testPoint[0]])
        score_testPoint = T.dot(entities[testPoint[1]], relations[testPoint[0]])
        rank = 1 + T.sum(scores > score_testPoint)
        return rank

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities,relations, seen_e2], sequences = [testData, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities,relations, testData, seen_e2], ranks,allow_input_downcast=True)
    return f

'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_e2[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def scoreMF(model, testData,seen_e2,MF_eval):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]


    ranks = MF_eval(entity_weights, relation_weights, testData, seen_e2)

    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)
