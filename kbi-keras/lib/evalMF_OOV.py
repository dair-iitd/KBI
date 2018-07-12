import numpy as np
import random
import sys
import theano 
import theano.tensor as T
from time import gmtime, strftime 
def ddict_set():
    return ddict(set)

def get_relations(testData, oov_flag=[]):
    #relation_map = ddict(set)
    #map(lambda test_fact, oov: relation_map[test_fact[0]].add((test_fact[1],oov)), testData, oov_flag)
    relation_map = ddict(ddict_set)
    map(lambda test_fact, oov: relation_map[test_fact[0]][test_fact[1]].add(oov), testData, oov_flag)

    return relation_map

def forward_pass_func(layers, inp):
    for layer in layers:
        W = layer
        inp = T.dot(inp, W) 
    return inp


def MF_oovEval_gen(): 
    entities      = T.dmatrix()
    entities_DM   = T.dmatrix()#
    relations     = T.dmatrix()
    Projected_all_e2    = T.dmatrix()#
    testData      = T.imatrix()
    testData_DM   = T.imatrix()#
    mapping_ep_e2 = T.ivector()##loc id == ep id && val at that loc is e2-id
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    seen_e2_DM = theano.typed_list.TypedListType(T.ivector)()

    # testPoint is relationId, entityPairId
    def MF_fn(testPoint, testPoint_DM, i, entities, entities_DM, relations, Projected_all_e2, seen_e2, seen_e2_DM, mapping_ep_e2):
        #all_ep_gen_vec = T.tanh(entities_DM[testPoint_DM[0]] + Projected_all_e2)
        all_ep_gen_vec = entities_DM[testPoint_DM[0]] + Projected_all_e2

        pre_trained_ep_index = mapping_ep_e2[seen_e2[i]]
        entities_trained = entities[seen_e2[i]]
        T.set_subtensor(all_ep_gen_vec[pre_trained_ep_index], entities_trained)
        
        scores_ep_gen   = T.dot(all_ep_gen_vec, relations[testPoint[0]])
        #scores = T.dot(entities[seen_e2[i]], relations[testPoint[0]])
        
        score_testPoint = T.dot(all_ep_gen_vec[testPoint_DM[2]], relations[testPoint[0]])
 
        #rank = 1 + T.sum(scores > score_testPoint)#
        rank = 1 + T.sum(scores_ep_gen > score_testPoint) - T.sum(scores_ep_gen[seen_e2_DM[i]] > score_testPoint)

        return rank

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities, entities_DM, relations, Projected_all_e2, seen_e2, seen_e2_DM, mapping_ep_e2], sequences = [testData, testData_DM, theano.tensor.arange(testData.shape[0])])

    f = theano.function([entities, entities_DM, relations, Projected_all_e2, testData, testData_DM, seen_e2, seen_e2_DM, mapping_ep_e2], ranks,allow_input_downcast=True)
    return f


'''
def MF_oovEval_gen():
    entities      = T.dmatrix()
    entities_DM   = T.dmatrix()#
    relations     = T.dmatrix()
    Project_e1    = T.dmatrix()#
    Project_e2    = T.dmatrix()#
    Project_b     = T.dvector()#
    testData      = T.imatrix()
    testData_DM   = T.imatrix()#
    mapping_ep_e2 = T.ivector()##loc id == ep id && val at that loc is e2-id
    seen_e2 = theano.typed_list.TypedListType(T.ivector)()
    seen_e2_DM = theano.typed_list.TypedListType(T.ivector)()
    oov_flags = T.ivector() # 1 if (e1,e2) is oov else 0

    def MF_fn(testPoint, testPoint_DM, oov_flag, i, entities, entities_DM, relations, Project_e1, Project_e2, Project_b, seen_e2, seen_e2_DM, mapping_ep_e2):
        e1_in           = entities_DM[testPoint_DM[0]]

        all_scores = T.dot(T.tanh(T.dot(Project_e1, e1_in) + T.dot(entities_DM, Project_e2) + Project_b), relations[testPoint[0]])

        score_testpoint_as_oov = all_scores[testPoint_DM[-1]]
        score_testpoint_as_trained = T.dot(relations[testPoint[0]], entities[testPoint[1]])
        score_testpoint = T.switch(oov_flag, score_testpoint_as_oov, score_testpoint_as_trained)
        
        e2s_seen_with_e1 = mapping_ep_e2[seen_e2[i]]
        all_scores_illegal = all_scores[e2s_seen_with_e1]

        # this can be 0
        rank_among_oovs = T.sum(all_scores > score_testpoint) - T.sum(all_scores_illegal > score_testpoint) - T.sum(all_scores[seen_e2_DM[i]] > score_testpoint)

        trained_e2_scores = T.dot(entities[seen_e2[i]], relations[testPoint[0]])
        # this can be 0 as well. 
        rank_among_non_oovs = T.sum(trained_e2_scores > score_testpoint)

        rank = 1 + rank_among_oovs + rank_among_non_oovs

        ##
        #same = T.sum(T.eq(scores_ep_gen, score_testPoint))
        #rank += same/2.0
        return rank

    ranks, ignore = theano.scan(MF_fn, non_sequences = [entities, entities_DM, relations, Project_e1, Project_e2, Project_b, seen_e2, seen_e2_DM, mapping_ep_e2], sequences = [testData, testData_DM, oov_flags, theano.tensor.arange(testData.shape[0])])
    
    f = theano.function([entities, entities_DM, relations, Project_e1, Project_e2, Project_b, testData, testData_DM, seen_e2, seen_e2_DM, mapping_ep_e2, oov_flags], ranks,allow_input_downcast=True)
    return f
'''


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
    print "OOV ranks"
    print_score(oov_ranks)
    print "Seen ranks"
    print_score(seen_ranks)   
    print np.mean(1.0/ranks), np.mean(ranks <= 10.0), np.mean(ranks <= 1.0), np.mean(ranks)
    if 1:
        def write_rank(data, file_name):
            f=open(file_name, "w")
            for i in data:
                f.write(str(i)+"\n")
            f.close()
        file_name = "ranks/allRank_"+opts.model+"_"+opts.dataset;file_name = file_name.replace("/", "_")
        #file_name = "ranks/%s_%s_ranks.txt" %(opts.model_path.split("_")[1],opts.model)
        write_rank(ranks, file_name.lower())

    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)

def scoreMF_OOV_generate_OOVep_embedding_fast(model, aux_model, opts, counter, testData, testData_DM, oov_flags, seen_e2, oov_flags_e2, seen_e2_DM, mapping_ep_e2, oov_counts, MF_oovEval_gen_fast):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            print "Obtained %s layer weights!!" %(layer.name)
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            print "Obtained %s layer weights!!" %(layer.name)
            relation_weights = layer.get_weights()[0]
    
    for layer in aux_model.layers:    
        if layer.name == 'dense_1':
            print "Obtained %s layer weights!!" %(layer.name)
            projection_weights = layer.get_weights()[0]
            projection_bias    = layer.get_weights()[1]
            print projection_weights[0],projection_bias[0]
        elif layer.name == 'entity_embeddings_DM':
            print "Obtained %s layer weights!!" %(layer.name)
            entity_weights_DM = layer.get_weights()[0]
            print entity_weights_DM[0]

    vect_dim          = entity_weights.shape[1]
    entity_weights    = entity_weights[:-1,]
    entity_weights_DM = entity_weights_DM[:-1,]

    #test_b = 1000

    entity_weights_DM_projected = np.dot(entity_weights_DM, projection_weights[:100,])
    Projected_all_e2            = np.dot(entity_weights_DM, projection_weights[100:,]) + projection_bias
    print "Ready!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    ranks  = np.array([]); batch_ = 6000
    if 0:
        num_iter = int(testData.shape[0]/batch_)
        print "Using batch size: %d" %batch_
        print "Number of iterations: %d" %num_iter
        start=0;end=start+batch_
        for test_b in xrange(num_iter):
            print test_b, strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ranks_n = MF_oovEval_gen_fast(entity_weights, entity_weights_DM_projected, relation_weights, Projected_all_e2, testData[start:end], testData_DM[start:end], seen_e2[start:end], seen_e2_DM[start:end], mapping_ep_e2)
            ranks = np.concatenate([ranks, ranks_n])
            start=end;end+=batch_
        ranks_n = MF_oovEval_gen_fast(entity_weights, entity_weights_DM_projected, relation_weights, Projected_all_e2, testData[start:end], testData_DM[start:end], seen_e2[start:end], seen_e2_DM[start:end], mapping_ep_e2)
        ranks = np.concatenate([ranks, ranks_n])
        assert ranks.shape[0] == testData.shape[0]
    else:
        #ranks = MF_oovEval_gen_fast(entity_weights, entity_weights_DM_projected, relation_weights, Projected_all_e2, testData[:test_b], testData_DM[:test_b], seen_e2[:test_b], seen_e2_DM[:test_b], mapping_ep_e2)
        ranks = MF_oovEval_gen_fast(entity_weights, entity_weights_DM_projected, relation_weights, Projected_all_e2 , testData, testData_DM, seen_e2, seen_e2_DM, mapping_ep_e2)    

    oov_ranks = np.array([rank for i, rank in enumerate(ranks) if oov_flags[i]])
    seen_ranks = np.array([rank for i, rank in enumerate(ranks) if not oov_flags[i]])
    print "OOV ranks"
    print_score(oov_ranks)
    print "Seen ranks"
    print_score(seen_ranks) 
    print np.mean(1.0/ranks), np.mean(ranks <= 10.0), np.mean(ranks <= 1.0), np.mean(ranks)
    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)


def scoreMF_OOV_generate_OOVep_embedding(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval, old_id, old_id_new_embeddings, new_id_new_embeddings, pretrained=False):
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

    ##
    if pretrained:
        old_id_new_embeddings  = entity_weights[old_id]
        new_id_new_embeddings  = entity_weights[new_id_new_embeddings]
    
    entity_weights[old_id] = old_id_new_embeddings 
    
    entity_weights         = np.concatenate([entity_weights[:-1,:], new_id_new_embeddings,entity_weights[-1,:].reshape(1,vect_dim)])

    ##    

    ranks = MF_oovEval(entity_weights, relation_weights, oov_embedding, testData, seen_e2, oov_counts, oov_flags)         
    oov_ranks = np.array([rank for i, rank in enumerate(ranks) if oov_flags[i]])
    seen_ranks = np.array([rank for i, rank in enumerate(ranks) if not oov_flags[i]])
    print "OOV ranks"
    print_score(oov_ranks)
    print "Seen ranks"
    print_score(seen_ranks)   
    print np.mean(1.0/ranks), np.mean(ranks <= 10.0), np.mean(ranks <= 1.0), np.mean(ranks)
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
    hits1 = np.mean(ranks <= 1.0)
    mr = np.mean(ranks)
    print("MRR: %5.4f, HITS@10: %5.4f, HITS@1: %5.4f, MR: %5.4f" %(mrr, hits, hits1, mr))

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
    print np.mean(1.0/ranks), np.mean(ranks <= 10.0), np.mean(ranks <= 1.0), np.mean(ranks)

    oov_ranks = np.array([rank for i, rank in enumerate(ranks) if oov_flags[i]])
    seen_ranks = np.array([rank for i, rank in enumerate(ranks) if not oov_flags[i]])
    print "OOV ranks"
    print_score(oov_ranks)
    print "Seen ranks"
    print_score(seen_ranks)

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
    print np.mean(1.0/ranks), np.mean(ranks <= 10.0), np.mean(ranks <= 1.0), np.mean(ranks)
    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)
