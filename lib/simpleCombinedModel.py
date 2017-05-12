from keras.layers import *
from lib.wrappers import *
from lib.MFmodel  import *
from lib.models   import *
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adagrad
from keras.layers.core import Lambda
from keras.models import Model
import theano
import theano.tensor as T
import numpy as np
import sys

def combine_model(X):
    r_dot_ep = X[0]
    r_dot_ep_prime = X[1]
    e1_cross_e2 = X[2]
    e1_cross_e2_prime = X[3]
    e1_prime_cross_e2 = X[4]
    relation_vectors = X[5]
    #weight_vector = X[6]
    
    '''MF'''
    denom_MF = r_dot_ep_prime 
    max_denom_MF = T.max(denom_MF, axis = 1, keepdims=True)
    denom_MF = T.exp(denom_MF - max_denom_MF).sum(axis=1)

    numer_MF = r_dot_ep 
    numer_MF = numer_MF - max_denom_MF.dimshuffle(0)
    net_score_MF= numer_MF - T.log(denom_MF) 
    
    '''DM'''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    denom_e1 = T.batched_dot(e1_cross_e2_prime, relation_vectors)
    max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
    denom_e1 = T.exp(denom_e1 - max_denom_e1).sum(axis=1)

    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = T.batched_dot(e1_prime_cross_e2, relation_vectors)
    max_denom_e2 = T.max(denom_e2, axis=1, keepdims=True)
    denom_e2 = T.exp(denom_e2 - max_denom_e2).sum(axis=1)

    numer = T.batched_dot(relation_vectors, e1_cross_e2)
    numer = 2*numer - max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
    net_score= numer - T.log(denom_e1) -T.log(denom_e2)
    
    '''combined score'''
    net_score += net_score_MF
    
    return -1*net_score

def combine_model_correct(X):
    r_dot_ep = X[0]
    r_dot_ep_prime = X[1]
    e1_cross_e2 = X[2]
    e1_cross_e2_prime = X[3]
    e1_prime_cross_e2 = X[4]
    relation_vectors = X[5]
    #weight_vector = X[6]
    
    '''MF'''
    denom_MF = r_dot_ep_prime 
    max_denom_MF = T.max(denom_MF, axis = 1, keepdims=True)
    denom_MF = T.exp(denom_MF).sum(axis=1)

    numer_MF = r_dot_ep 
    numer_MF = numer_MF 
    net_score_MF= numer_MF - T.log(denom_MF) 
    
    '''DM'''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    denom_e1 = T.batched_dot(e1_cross_e2_prime, relation_vectors)
    max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
    denom_e1 = T.exp(denom_e1).sum(axis=1)

    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = T.batched_dot(e1_prime_cross_e2, relation_vectors)
    max_denom_e2 = T.max(denom_e2, axis=1, keepdims=True)
    denom_e2 = T.exp(denom_e2).sum(axis=1)

    numer = T.batched_dot(relation_vectors, e1_cross_e2)
    numer = 2*numer #- max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
    net_score= numer - T.log(denom_e1) -T.log(denom_e2)
    
    '''combined score'''
    net_score += net_score_MF
    
    return -1*net_score


def getMF_score(kb_entity_pairs, kb_relations, relation_vectors, neg_samples_kb, opts): 
    vect_dim      = opts.vect_dim
    neg_samples   = opts.neg_samples
    num_entity_pairs  = opts.num_entity_pairs
    num_relations = opts.num_relations
    l2_reg_entity_pair = opts.l2_entity
    l2_reg_relation = opts.l2_relation    

    entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs+1, init='normal', name = 'entity_embeddings_MF',  W_regularizer=l2(l2_reg_entity_pair))
    
    entity_pair_vectors = Flatten()(entity_pairs(kb_entity_pairs))
    entity_pair_negative_vectors = entity_pairs(neg_samples_kb)



    r_dot_e = merge([relation_vectors, entity_pair_vectors], mode = lambda X: T.batched_dot(X[0], X[1]), output_shape = (None,))
    r_dot_e_prime = merge([relation_vectors, entity_pair_negative_vectors], mode ='dot', output_shape = (None, neg_samples), dot_axes=(1,2))

    return r_dot_e, r_dot_e_prime

def buildMF_plus_distMult(opts):
    neg_samples = opts.neg_samples

    kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    kb_entity_pairs = Input(shape=(1,), dtype='int32', name='kb_entity_pairs')
    neg_samples_kb_MF  = Input(shape=(neg_samples, ), dtype = 'int32', name='kb_neg_examples_entity_pairs')
    neg_samples_kb_DM  = Input(shape=(2*neg_samples,), dtype = 'int32', name='kb_neg_examples')
    
    '''
    weights = Embedding(output_dim=1, input_dim=2, init='normal', name='combination parameters')
    weight_vector = weights(T.arange(2))
    '''
    
    e1_cross_e2, e1_cross_e2_prime, e1_prime_cross_e2, relation_vectors  = getDM_score(kb_entities, kb_relations, neg_samples_kb_DM, opts)


    if not opts.shared_r:
        relations = Embedding(output_dim=opts.vect_dim, input_dim=opts.num_relations, input_length=1,init='normal', name='relation_embeddings_MF', W_regularizer=l2(opts.l2_relation))
        relation_vectors_MF = Flatten()(relations(kb_relations))
    else:
        relation_vectors_MF = relation_vectors

    r_dot_ep, r_dot_ep_prime = getMF_score(kb_entity_pairs, kb_relations, relation_vectors_MF, neg_samples_kb_MF, opts)

    #get_score = merge([r_dot_ep, r_dot_ep_prime, e1_cross_e2, e1_cross_e2_prime, e1_prime_cross_e2, relation_vectors, weight_vector], mode=combine_model, output_shape=(None,1))
    log_likelihood_kb = merge([r_dot_ep, r_dot_ep_prime, e1_cross_e2, e1_cross_e2_prime, e1_prime_cross_e2, relation_vectors], mode=combine_model, output_shape=(None,1))

    alg = get_optimizer(opts)

    model = Model(input=[kb_entities, kb_entity_pairs, kb_relations, neg_samples_kb_DM, neg_samples_kb_MF], output=log_likelihood_kb)
    model.compile(loss = lossFn, optimizer=alg)
    
    return model



