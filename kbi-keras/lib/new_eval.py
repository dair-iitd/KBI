import numpy as np
import theano
import theano.typed_list
import theano.tensor as T
from collections import defaultdict as ddict

''' Theano functions for scoring models'''

def distMult():
    entities  = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    triples_known = theano.typed_list.TypedListType(T.ivector)()
    def distMult_fn(testPoint, i, entities, relations, triples_known):
        e1_cross_e2_all = entities[testPoint[0]]*entities
        distMult_scores = T.dot(e1_cross_e2_all, relations[testPoint[1]])
        raw_rank = 1 + T.sum(distMult_scores > distMult_scores[testPoint[-1]])
        filtered_rank = raw_rank - T.sum(distMult_scores[triples_known[i]] > distMult_scores[testPoint[-1]])
        return raw_rank, filtered_rank
    ranks, ignore = theano.scan(distMult_fn, non_sequences = [entities, relations, triples_known], sequences = [testData, theano.tensor.arange(testData.shape[0])])
    f = theano.function([entities, relations, testData, triples_known], ranks,allow_input_downcast=True)
    return f

def create_EplusDMfn(alpha):
    def EplusDistMult():
        entities   = T.dmatrix()
        entities_E = T.dmatrix()
        relations = T.dmatrix()
        relations_o = T.dmatrix()
        testData = T.imatrix()
        triples_known = theano.typed_list.TypedListType(T.ivector)()
        def EplusDistMult_fn(testPoint, i, entities, entities_E, relations, relations_o, triples_known):
            e1_cross_e2_all = entities[testPoint[0]]*entities
            distMult_scores = T.dot(e1_cross_e2_all, relations[testPoint[1]])
            E_scores = T.dot(entities_E, relations_o[testPoint[1]])
            scores = alpha*distMult_scores + (1-alpha)*E_scores
            raw_rank = 1 + T.sum(scores > scores[testPoint[-1]])
            filtered_rank = raw_rank - T.sum(scores[triples_known[i]] > scores[testPoint[-1]])
            return raw_rank, filtered_rank

        ranks, ignore = theano.scan(EplusDistMult_fn, non_sequences = [entities, entities_E, relations,relations_o, triples_known], sequences = [testData, theano.tensor.arange(testData.shape[0], dtype='int64')])
        f = theano.function([entities, entities_E, relations, relations_o, testData, triples_known], ranks,allow_input_downcast=True)
        return f

    return EplusDistMult

def E():
    entities = T.dmatrix()
    relations = T.dmatrix()
    testData = T.imatrix()
    triples_known = theano.typed_list.TypedListType(T.ivector)()
    def E_fn(testPoint, i, entities, relations, triples_known):
        scores = T.dot(entities, relations[testPoint[1]])
        raw_rank = 1 + T.sum( scores > scores[testPoint[-1]])
        filtered_rank = raw_rank - T.sum(scores[triples_known[i]] > scores[testPoint[-1]])
        return raw_rank, filtered_rank

    ranks, ignore = theano.scan(E_fn, non_sequences = [entities,relations,triples_known], sequences = [testData, theano.tensor.arange(testData.shape[0], dtype='int64')])
    f = theano.function([entities,relations,testData, triples_known], ranks,allow_input_downcast=True)
    return f

def deepDistMult():
    entities = T.dmatrix()
    relations = T.dmatrix()
    projection_matrix = T.dmatrix()
    testData  = T.imatrix()
    triples_known = theano.typed_list.TypedListType(T.ivector)() 
    def deepDistMult_fn(testPoint, i, entities, relations, projection_matrix, triples_known):
        e1 = T.tile(entities[testPoint[0]].dimshuffle('x', 0), (entities.shape[0], 1))
        e1_concat_e2_all_transformed = T.tanh(T.dot(T.concatenate([e1, entities], axis=-1), projection_matrix))
        scores = T.dot(e1_concat_e2_all_transformed, relations[testPoint[1]])   
        raw_rank = 1 + T.sum( scores > scores[testPoint[-1]])
        filtered_rank = raw_rank - T.sum(scores[triples_known[i]] > scores[testPoint[-1]])
        return raw_rank, filtered_rank
    
    ranks, ignore = theano.scan(deepDistMult_fn, non_sequences = [entities,relations, projection_matrix,triples_known], sequences = [testData, theano.tensor.arange(testData.shape[0], dtype='int64')])
    f = theano.function([entities,relations, projection_matrix, testData, triples_known], ranks,allow_input_downcast=True)
    return f


def naiveDeepDistMult():
    entities = T.dmatrix()
    relations = T.dmatrix()
    projection_matrix = T.dmatrix()
    testData  = T.imatrix()
    triples_known = theano.typed_list.TypedListType(T.ivector)() 
    def deepDistMult_fn(testPoint, i, entities, relations, projection_matrix, triples_known):
        e1_cross_e2_all = entities[testPoint[0]]*entities           
        e1_cross_e2_all_transformed = T.tanh(T.dot(e1_cross_e2_all, projection_matrix))
        scores = T.dot(e1_cross_e2_all_transformed, relations[testPoint[1]])    
        raw_rank = 1 + T.sum( scores > scores[testPoint[-1]])
        filtered_rank = raw_rank - T.sum(scores[triples_known[i]] > scores[testPoint[-1]])
        return raw_rank, filtered_rank
    
    ranks, ignore = theano.scan(deepDistMult_fn, non_sequences = [entities,relations, projection_matrix,triples_known], sequences = [testData, theano.tensor.arange(testData.shape[0], dtype='int64')])
    f = theano.function([entities,relations, projection_matrix, testData, triples_known], ranks,allow_input_downcast=True)
    return f


def getRelationScores(all_ranks, testData):
    macro_ranks = ddict(list)
    for i, testPoint in enumerate(testData):
        e1, r, e2 = testPoint
        if (all_ranks[i] == 0):
            print("incorrect at %d" %i)
        macro_ranks[r].append(all_ranks[i]) 

    scoresAllRelations  = {}
    for relation in macro_ranks:
        scores = macro_ranks[relation]
        mrr_relation=0.0
        hits_relation=0.0
        for score in scores:
            mrr_relation += 1.0/score
            hits_relation += (score <= 10)  

        mrr_relation /= len(scores)
        hits_relation /= len(scores)
        scoresAllRelations[relation]  = [mrr_relation, hits_relation]

    return scoresAllRelations
        
def getMacroScores(all_ranks, testData):
    macro_ranks = ddict(list)
    for i, testPoint in enumerate(testData):
        e1, r, e2 = testPoint
        macro_ranks[(r,e2)].append(all_ranks[i])

    macro_mrr=0.0
    macro_hits=0.0
    for r_e2Pair in macro_ranks:
        scores = macro_ranks[r_e2Pair]
        mrr_pair=0.0
        hits_pair=0.0
        for score in scores:
            mrr_pair += 1.0/score
            hits_pair += (score <= 10)  

        mrr_pair /= len(scores)
        hits_pair /= len(scores)
        macro_mrr += mrr_pair   
        macro_hits += hits_pair


    macro_mrr /= len(macro_ranks)
    macro_hits /= len(macro_ranks)
    return macro_mrr, macro_hits


def getStats(all_ranks, testData):
    relation_scores = getRelationScores(all_ranks[1], testData)
    raw_ranks = all_ranks[0]; ranks = all_ranks[1]
    mrr = np.mean(1.0/ranks); raw_mrr = np.mean(1.0/raw_ranks)
    hits10 = np.mean(ranks <= 10); raw_hits10 = np.mean(raw_ranks <= 10)
    #return relation_scores,[mrr, hits10], [raw_mrr, raw_hits10]
    return mrr, hits10
def scoreDistMult(model, testData,triples_known, distMult_fn):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
    
    all_ranks = distMult_fn(entity_weights, relation_weights, testData, triples_known)  
    return getStats(all_ranks, testData)


def scoreE(model, testData,triples_known, E_fn):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights = layer.get_weights()[0]
    
    all_ranks = E_fn(entity_weights, relation_weights, testData, triples_known)     
    return getStats(all_ranks, testData)

def scoreDeepDistMult(model, testData,triples_known, deepDistMult_fn):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'dense_transform':
            projection_matrix= layer.get_weights()[0]
    
    all_ranks = deepDistMult_fn(entity_weights, relation_weights, projection_matrix, testData, triples_known)   
    return getStats(all_ranks, testData)


def scoreEplusDistMult(model, testData, triples_known, EplusDistMult_fn):
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings_E':
            entity_weights_E = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings_o':
            relation_weights_o = layer.get_weights()[0]
        elif layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
    
    all_ranks  = EplusDistMult_fn(entity_weights, entity_weights_E, relation_weights, relation_weights_o, testData, triples_known)
    return getStats(all_ranks, testData)    
