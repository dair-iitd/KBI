import numpy as np
import random
import sys
import theano 
import theano.tensor as T
from keras.layers import *

def forward_pass_func(layers, inp):
    for layer in layers[:-1]:
        W, b = layer
        inp = T.tanh(T.dot(inp, W) + b)

    W, b = layers[-1]
    return T.nnet.sigmoid(T.dot(inp, W) + b)

def get_input(MF_scores, DM_scores, aux_features):
    aux_features_tiled = T.tile(aux_features.dimshuffle('x',0), (MF_scores.shape[0], 1))
    return T.concatenate([MF_scores.dimshuffle(0,'x'), DM_scores.dimshuffle(0,'x'), aux_features_tiled], axis=1)

def get_featureNet2_scores(layers, aux_features_curr, scores_set1, scores_set2, scores_set3, score_testPoint):
    score_testPoint_MF, score_testPoint_DM = score_testPoint
    scores_MF, scores_DM = scores_set1
    score_oov_MF_vector, scores_DM_set2 = scores_set2
    score_oov_MF, score_oov_DM = scores_set3
    score_oov_MF_vector = T.tile(score_oov_MF_vector, scores_DM_set2.shape[0])

    f1 = forward_pass_func(layers, aux_features_curr)[0]
    #f2 = forward_pass_func(layers, T.tile(aux_features_curr.dimshuffle('x',0), (scores_DM.shape[0], 1)))[:,0]
    #f3 = forward_pass_func(layers,T.tile(aux_features_curr.dimshuffle('x',0), (scores_DM_set2.shape[0], 1)))[:, 0]

    score_testPoint  = f1*score_testPoint_MF + (1.0 - f1)*score_testPoint_DM
    scores_set1 = f1*scores_MF + (1.0 - f1)*scores_DM
    #scores_set1 = scores_MF + scores_DM
    scores_set2 = f1*score_oov_MF_vector + (1.0 - f1)*scores_DM_set2
    #scores_set2 = score_oov_MF + scores_DM_set2
    scores_set3 = f1**score_oov_MF + (1.0 - f1)*score_oov_DM
    #scores_set3 = score_oov_MF + score_oov_DM

    return score_testPoint, scores_set1, scores_set2, scores_set3, f1

def get_featureNet_scores(layers, aux_features_curr, scores_set1, scores_set2, scores_set3, score_testPoint):
    score_testPoint_MF, score_testPoint_DM = score_testPoint
    scores_MF, scores_DM = scores_set1
    score_oov_MF_vector, scores_DM_set2 = scores_set2
    score_oov_MF, score_oov_DM = scores_set3
    score_oov_MF_vector = T.tile(score_oov_MF_vector, scores_DM_set2.shape[0])

    f1 = forward_pass_func(layers, aux_features_curr)[0]
    score_testPoint  = f1*score_testPoint_MF + score_testPoint_DM
    scores_set1 = f1*scores_MF + scores_DM#ep seen
    scores_set2 = f1*score_oov_MF + scores_DM_set2#ep not seen but e's are seen
    scores_set3 = f1*score_oov_MF + score_oov_DM#both es aren't seen

    return score_testPoint, scores_set1, scores_set2, scores_set3, f1

def get_normalized_scores(layers, scores):    
    W, b = layers[0]
    inp = T.switch(T.sum(scores),scores*W[0][0] + b[0]  ,scores)
    return inp


def get_joint_scores(layers, aux_features_curr, scores_set1, scores_set2, scores_set3, score_testPoint):
    score_testPoint_MF, score_testPoint_DM = score_testPoint
    scores_MF, scores_DM = scores_set1
    score_oov_MF_vector, scores_DM_set2 = scores_set2
    score_oov_MF, score_oov_DM = scores_set3

    stack_scores = T.stack([score_testPoint_MF, score_testPoint_DM])
    stack_oov_scores = T.stack([score_oov_MF, score_oov_DM])
    oov_point_input  = T.concatenate([stack_oov_scores, aux_features_curr])
    test_point_input = T.concatenate([stack_scores, aux_features_curr])
    score_testPoint  = forward_pass_func(layers, test_point_input)[0]

    scores_set1 = forward_pass_func(layers, get_input(scores_MF, scores_DM,aux_features_curr))
    scores_set2 = forward_pass_func(layers, get_input(T.tile(score_oov_MF_vector, scores_DM_set2.shape[0]), scores_DM_set2, aux_features_curr))
    scores_set3 = forward_pass_func(layers, oov_point_input)[0]

    return score_testPoint, scores_set1, scores_set2, scores_set3

def get_data_stats(all_scores):
    mean = T.switch(all_scores.shape[0],T.mean(all_scores),0)#T.switch(T.sum(all_scores),T.mean(all_scores),0)
    std  = T.switch(all_scores.shape[0],T.std(all_scores),1)#T.switch(T.sum(all_scores),T.std(all_scores),1)
    std  = T.switch(std,std,1)
    return mean, std#,T.max(all_scores),T.min(all_scores)

def normalize_data(scores, mean, std):
    return T.switch(T.sum(scores),(scores-mean)/std,scores)

def model_eval(get_scores):
    entityPairs  = T.fmatrix()
    entities  = T.fmatrix()
    relations = T.fmatrix()
    testData_DM = T.imatrix()
    testData_MF = T.imatrix()
    entity_oov_embedding = T.fvector()
    entityPair_oov_embedding = T.fvector()
    normalize_eval = T.iscalar()
    normalize = T.iscalar()
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
    
    nnet_W1 = T.fmatrix()
    nnet_W2 = T.fmatrix()
    nnet_W3 = T.fmatrix()

    nnet_b1 = T.fvector()
    nnet_b2 = T.fvector()
    nnet_b3 = T.fvector()
    aux_features = T.fmatrix()

    layers = [(nnet_W1, nnet_b1), (nnet_W2, nnet_b2), (nnet_W3, nnet_b3)]

    normalize_DM_W1 = T.fmatrix()
    normalize_DM_b1 = T.fvector()
    normalize_MF_W1 = T.fmatrix()
    normalize_MF_b1 = T.fvector()

    layers_normalize_DM = [(normalize_DM_W1, normalize_DM_b1)]
    layers_normalize_MF = [(normalize_MF_W1, normalize_MF_b1)]

    def MF_fn(testPoint_DM, testPoint_MF, i, oov_flag_e1, oov_flag_e2, oov_flag, entityPairs, entities, relations, entityPair_oov_embedding, entity_oov_embedding, allowed_entityPair, set1_e2, set2_e2, set3_e2, normalize_eval, normalize):
        # score of allowed e2s
        scores_MF = T.tanh(T.dot(entityPairs[allowed_entityPair[i]], relations[testPoint_MF[0]]))
        #scores_MF = T.dot(entityPairs[allowed_entityPair[i]], relations[testPoint_MF[0]])
        # score for oov (e1,e2)s 
        score_oov_MF = T.tanh(T.dot(entityPair_oov_embedding,relations[testPoint_MF[0]])) 
        #score_oov_MF = T.dot(entityPair_oov_embedding,relations[testPoint_MF[0]]) 
        score_nonOOV_MF = T.tanh(T.dot(entityPairs[testPoint_MF[1]], relations[testPoint_MF[0]]))
        #score_nonOOV_MF = T.dot(entityPairs[testPoint_MF[1]], relations[testPoint_MF[0]])

        # based on whether (e1,e2) is OOV pick the score for the current testPoint
        score_testPoint_MF = T.switch(oov_flag, score_oov_MF, score_nonOOV_MF)
        
        e1_fact_embedding = T.switch(oov_flag_e1, entity_oov_embedding, entities[testPoint_DM[0]])
        e2_fact_embedding = T.switch(oov_flag_e2, entity_oov_embedding, entities[testPoint_DM[2]])

        # score of allowed e2s -> (e1,e2) seen -> e2 seen
        scores_DM   = T.tanh(T.dot(e1_fact_embedding*entities[set1_e2[i]], relations[testPoint_DM[1]]))
        #scores_DM   = T.dot(e1_fact_embedding*entities[set1_e2[i]], relations[testPoint_DM[1]])
        # score for the test point
        score_testPoint_DM  = T.tanh(T.dot(relations[testPoint_DM[1]], e1_fact_embedding*e2_fact_embedding))
        #score_testPoint_DM  = T.dot(relations[testPoint_DM[1]], e1_fact_embedding*e2_fact_embedding)
        score_oov_DM = T.tanh(T.dot(relations[testPoint_DM[1]], e1_fact_embedding*entity_oov_embedding))
        #score_oov_DM = T.dot(relations[testPoint_DM[1]], e1_fact_embedding*entity_oov_embedding)
        
        # score for e2s such that (e1,e2) non seen but e2 non OOV.
        scores_DM_set2 = T.tanh(T.dot(e1_fact_embedding*entities[set2_e2[i]], relations[testPoint_DM[1]]))
        #scores_DM_set2 = T.dot(e1_fact_embedding*entities[set2_e2[i]], relations[testPoint_DM[1]])
        
        
        #Normalize scores using pretrained weights
        scores_MF = T.switch(normalize, get_normalized_scores(layers_normalize_MF, scores_MF), scores_MF)
        score_testPoint_MF = T.switch(normalize, get_normalized_scores(layers_normalize_MF, score_testPoint_MF), score_testPoint_MF)
        score_oov_MF = T.switch(normalize, get_normalized_scores(layers_normalize_MF, score_oov_MF), score_oov_MF)
        scores_DM = T.switch(normalize, get_normalized_scores(layers_normalize_DM, scores_DM), scores_DM)
        score_testPoint_DM = T.switch(normalize, get_normalized_scores(layers_normalize_DM, score_testPoint_DM), score_testPoint_DM)
        score_oov_DM = T.switch(normalize, get_normalized_scores(layers_normalize_DM, score_oov_DM), score_oov_DM)
        scores_DM_set2 = T.switch(normalize, get_normalized_scores(layers_normalize_DM, scores_DM_set2), scores_DM_set2)
        
        #DM and MF score normalization
        
        mean_DM, std_DM = get_data_stats(T.concatenate([scores_DM,scores_DM_set2, T.stack([score_oov_DM])]))
        scores_DM = T.switch(normalize_eval, normalize_data(scores_DM, mean_DM, std_DM), scores_DM)
        mean_MF, std_MF = get_data_stats(T.concatenate([scores_MF, T.stack([score_oov_MF])]))
        scores_MF = T.switch(normalize_eval, normalize_data(scores_MF, mean_MF, std_MF), scores_MF) 
        score_oov_DM = T.switch(normalize_eval,normalize_data(score_oov_DM, mean_DM, std_DM),score_oov_DM)
        score_oov_MF = T.switch(normalize_eval,normalize_data(score_oov_MF, mean_MF, std_MF), score_oov_MF)
        score_testPoint_MF =T.switch(normalize_eval,(score_testPoint_MF - mean_MF)/std_MF,score_testPoint_MF)
        score_testPoint_DM =T.switch(normalize_eval, (score_testPoint_DM - mean_DM)/std_DM, score_testPoint_DM)
        #
        
        score_testPoint, scores_set1, scores_set2, scores_set3, f1 = get_scores(layers, aux_features[i], [scores_MF, scores_DM], [T.stack(score_oov_MF), scores_DM_set2], [score_oov_MF, score_oov_DM], [score_testPoint_MF, score_testPoint_DM])

        rank = 1 + T.sum(scores_set1 > score_testPoint) + T.sum(scores_set2 > score_testPoint)
        oov_comparison = score_testPoint < scores_set3
        rank = T.switch(oov_comparison, rank + set3_e2[i], rank)
        rank = T.switch(oov_flag_e2, rank + (set3_e2[i]/2.0), rank)         

        same = T.sum(T.eq(scores_set1, score_testPoint)) + T.sum(T.eq(scores_set2,score_testPoint))

        rank += same/2.0
        
        same = same/(scores_set1.shape[0] + scores_set2.shape[0]*1.0)
        '''
        dataStats = T.concatenate([get_data_stats(T.concatenate([scores_set1,scores_set2])),
        get_data_stats(scores_MF),
        get_data_stats(T.concatenate([scores_DM,scores_DM_set2]))])
        oov_scores = T.stack([score_oov_MF, score_oov_DM])
            
        return rank, f1, score_testPoint_DM, score_testPoint_MF, dataStats, oov_scores
        '''
        return rank, f1, score_testPoint_DM, score_testPoint_MF, same*100.0                                                 
    ranks, ignore = theano.scan(MF_fn, non_sequences = [entityPairs, entities,relations, entityPair_oov_embedding, entity_oov_embedding, allowedEP_MF, set1_e2, set2_e2, set3_e2, normalize_eval, normalize], sequences = [testData_DM, testData_MF, theano.tensor.arange(testData_DM.shape[0]), oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF])                                                        
    f = theano.function([normalize_eval, normalize, entityPairs, entities, relations, entityPair_oov_embedding, entity_oov_embedding, testData_DM, testData_MF, allowedEP_MF, set1_e2, set2_e2, oov_flag_e1_DM, oov_flag_e2_DM, oov_flags_MF, set3_e2, aux_features, nnet_W1, nnet_b1, nnet_W2, nnet_b2, nnet_W3, nnet_b3, normalize_DM_W1, normalize_DM_b1, normalize_MF_W1, normalize_MF_b1], ranks, allow_input_downcast=True)
    
    return f

def joint_eval():
    return model_eval(get_joint_scores)

def featureNet_eval():
    return model_eval(get_featureNet_scores)

def featureNet2_eval():
    return model_eval(get_featureNet2_scores)


'''
    seen_e2[i] = all (e1,e2) ids such that they are seen in train
    oov_counts[i] all e2s such that they are not OOV. (includes non OOVs and e2s removed via filtered measures)
'''
def score_joint(model, opts, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM, aux_features, eval_func):

    nnet_layers = {} 
    normalize_weights_DM = {}
    normalize_weights_MF = {}
    print("evaluating.")
    for layer in model.layers:
        if layer.name == 'entity_embeddings_MF':
            entityPair_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
        elif layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name.startswith('dense'):
            curr_layer = layer.get_weights()
            layer_number = int(layer.name.split('_')[1])
            nnet_layers['W%d' %layer_number] = curr_layer[0]
            nnet_layers['b%d' %layer_number] = curr_layer[1]
        elif layer.name.startswith('normalize_DM'):
            curr_layer = layer.get_weights()
            normalize_weights_DM['W1'] = curr_layer[0]
            normalize_weights_DM['b1'] = curr_layer[1]
            print normalize_weights_DM
        elif layer.name.startswith('normalize_MF'):
            curr_layer = layer.get_weights()
            normalize_weights_MF['W1'] = curr_layer[0]
            normalize_weights_MF['b1'] = curr_layer[1]
            print normalize_weights_MF

    oov_embedding_ep = entityPair_weights[-1]
    oov_embedding_e = entity_weights[-1] 

    if opts.normalize_score_eval:
        print "normalize scores only at eval!"
    if opts.normalize_score:
        print "normalize scores using trained bias and slope!"
        ranks, alphas, DM, MF, same = eval_func(opts.normalize_score_eval, opts.normalize_score, entityPair_weights, entity_weights, relation_weights, oov_embedding_ep, oov_embedding_e, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM, aux_features, nnet_layers['W1'], nnet_layers['b1'], nnet_layers['W2'], nnet_layers['b2'], nnet_layers['W3'], nnet_layers['b3'], normalize_weights_DM['W1'], normalize_weights_DM['b1'], normalize_weights_MF['W1'], normalize_weights_MF['b1'])
    else:
        ranks, alphas, DM, MF, same = eval_func(opts.normalize_score_eval, opts.normalize_score, entityPair_weights, entity_weights, relation_weights, oov_embedding_ep, oov_embedding_e, testData_DM, testData_MF, allowedEP, allowedEP_e2_DM, notAllowedEP_e2_DM, oov_flags_e1_DM, oov_flags_e2_DM, oov_flags_MF, oov_e2_count_DM, aux_features, nnet_layers['W1'], nnet_layers['b1'], nnet_layers['W2'], nnet_layers['b2'], nnet_layers['W3'], nnet_layers['b3'], np.array([[0]], dtype='float32'), np.array([0], dtype='float32'),np.array([[0]], dtype='float32'), np.array([0], dtype='float32'))


    mrr = np.mean(1.0/ranks)
    hits = np.mean(ranks <= 10.0)

    layers = [(nnet_layers['W1'], nnet_layers['b1']), (nnet_layers['W2'], nnet_layers['b2']), (nnet_layers['W3'], nnet_layers['b3'])] 
    
    print "(DM) max: %5.4f min: %5.4f, mean: %5.4f, std-dev: %5.4f" %(max(DM), min(DM), np.mean(DM), np.std(DM))
    print "(MF) max: %5.4f min: %5.4f, mean: %5.4f, std-dev: %5.4f" %(max(MF),min(MF), np.mean(MF), np.std(MF))
    print "(alphas) max: %5.4f min: %5.4f, mean: %5.4f, std-dev: %5.4f" %(max(alphas),min(alphas), np.mean(alphas), np.std(alphas))
    print "(same) max: %5.4f min: %5.4f, mean: %5.4f, std-dev: %5.4f" %(max(same), min(same), np.mean(same), np.std(same))
   
    return np.mean(1.0/ranks), np.mean(ranks <= 10.0)


