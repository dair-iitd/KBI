from keras.layers import *
from lib.wrappers import *
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adagrad
from keras.layers.core import Lambda
from keras.models import Model
import theano
import theano.tensor as T
import numpy as np
import sys
from keras.constraints import nonneg
    

''' Some helper functions '''

def custom_fn_MF(input):
    r_dot_e   = input[0]
    r_dot_e_prime= input[1]
    
    '''denom_e1 = sum{j=1..200}dot(r_s, e1) + dot(r_o, e2j') '''
    denom_e = r_dot_e_prime 
    max_denom_e = T.max(denom_e, axis = 1, keepdims=True)
    denom_e = T.exp(denom_e - max_denom_e).sum(axis=1)

    numer = r_dot_e 
    numer = numer - max_denom_e.dimshuffle(0)
    net_score= numer - T.log(denom_e) 
    return -1*net_score

def custom_fn_MF_max_margin(input):
    r_dot_e   = input[0]
    r_dot_e_prime= input[1]
    denom_e = r_dot_e_prime 
    numer = r_dot_e 
    net_score= T.sum(T.maximum(0,1.0 + denom_e.dimshuffle(0,1) - numer.dimshuffle(0,'x') ) )
    return 1*net_score


def lossFn(y_true, y_pred):
    return T.mean(y_pred)


def getMF_score(kb_entity_pairs, kb_relations, neg_samples_kb, opts): 
    vect_dim      = opts.vect_dim
    neg_samples   = opts.neg_samples
    num_entity_pairs  = opts.num_entity_pairs
    num_relations = opts.num_relations
    l2_reg_entity_pair = opts.l2_entity
    l2_reg_relation = opts.l2_relation    

    if opts.oov_train:
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs+1, init='normal', name = 'entity_embeddings',  W_regularizer=l2(l2_reg_entity_pair))
    else:
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs, init='normal', name = 'entity_embeddings',  W_regularizer=l2(l2_reg_entity_pair))

    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg_relation))

    entity_pair_vectors = Flatten()(entity_pairs(kb_entity_pairs))
    entity_pair_negative_vectors = entity_pairs(neg_samples_kb)

    relation_vectors = Flatten()(relations(kb_relations))
    r_dot_e = merge([relation_vectors, entity_pair_vectors], mode = lambda X: T.batched_dot(X[0], X[1]), output_shape = ())
    r_dot_e_prime = merge([relation_vectors, entity_pair_negative_vectors], mode ='dot', output_shape = (neg_samples), dot_axes=(1,2))

    return r_dot_e, r_dot_e_prime
''' Energy based neural net models '''           

def build_MFModel(opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_relations = opts.num_relations
    warm_start    = opts.warm_start
    optimizer     = opts.optimizer
    num_entity_pairs  = opts.num_entity_pairs
    l2_reg_entity_pair = opts.l2_entity
    l2_reg_relation = opts.l2_relation    

    #define all inputs
    kb_entity_pairs = Input(shape=(1,), dtype='int32', name='kb_entity_pairs')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    neg_samples_kb  = Input(shape=(neg_samples, ), dtype = 'int32', name='kb_neg_examples')
   

    r_dot_e, r_dot_e_prime = getMF_score(kb_entity_pairs, kb_relations, neg_samples_kb, opts) 
    if opts.loss == "ll":
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF, output_shape = (1,))
    elif opts.loss == "mm":
        print "Using Max-margin loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_max_margin, output_shape = (1,))
        
    if optimizer=='Adagrad':
        alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)

    model = Model(input=[kb_entity_pairs, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    return model





