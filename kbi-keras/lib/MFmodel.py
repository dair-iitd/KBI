import keras
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
def get_softmax_approx(input):
    score, score_e1_corrupt, score_e2_corrupt = input
    ''' f(e1, r, e2) = r.T(e1*e2) '''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    max_denom_e1 = T.max(score_e1_corrupt, axis = 1, keepdims=True)
    denom_e1 = T.exp(score_e1_corrupt - max_denom_e1).sum(axis=1)

    max_denom_e2 = T.max(score_e2_corrupt, axis = 1, keepdims=True)
    denom_e2 = T.exp(score_e2_corrupt - max_denom_e2).sum(axis=1)

    numer = 2*score - max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
    net_score= numer - T.log(denom_e1) - T.log(denom_e2)
    return -1*net_score


def get_logistic(input):
    print "I am here! - get_logistic",len(input)
    if len(input) ==3:
        score, score_e1_corrupt, score_e2_corrupt = input
    else:
        score, score_e1_corrupt = input
    loss_score = T.nnet.softplus(-1*score)
    loss_e1_corrupt = T.sum(T.nnet.softplus(1*score_e1_corrupt), axis=1)

    if len(input) ==3:
        loss_e2_corrupt = T.sum(T.nnet.softplus(1*score_e2_corrupt), axis=1)
        net_loss = loss_score + loss_e1_corrupt + loss_e2_corrupt
        net_loss = net_loss/(1+(2)*score_e1_corrupt.shape[1])
    else:
        net_loss = loss_score + loss_e1_corrupt
        net_loss = net_loss/(1+(1)*score_e1_corrupt.shape[1])
    return 1*net_loss


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

def custom_fn_MF_logistic(input):
    r_dot_e,r_dot_e_prime = input
    loss_ep_corrupt = T.sum(T.nnet.softplus(1*r_dot_e_prime), axis=1)
    loss_score = T.nnet.softplus(-1*r_dot_e)
    net_loss = loss_ep_corrupt + loss_score
    net_loss = net_loss/(1+r_dot_e_prime.shape[1])
    return 1*net_loss

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
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs+1, init='normal', name = 'entity_embeddings')#,  W_regularizer=l2(l2_reg_entity_pair))
    else:
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs, init='normal', name = 'entity_embeddings')#,  W_regularizer=l2(l2_reg_entity_pair))

    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')#, W_regularizer=l2(l2_reg_relation))

    entity_pair_vectors = Flatten()(entity_pairs(kb_entity_pairs))
    entity_pair_negative_vectors = entity_pairs(neg_samples_kb)

    relation_vectors = Flatten()(relations(kb_relations))
    r_dot_e = merge([relation_vectors, entity_pair_vectors], mode = lambda X: T.batched_dot(X[0], X[1]), output_shape = ())
    r_dot_e_prime = merge([relation_vectors, entity_pair_negative_vectors], mode ='dot', output_shape = (neg_samples), dot_axes=(1,2))

    def l2_theo(input):
        entity_vectors, relation_vectors, entity_negative_vectors = input
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )
        return (T.sqr(entity_vectors).sum()/denom) + (T.sqr(entity_negative_vectors).sum()/denom) + (T.sqr(relation_vectors).mean()) 

    if opts.theo_reg:
        reg = merge([entity_pair_vectors, relation_vectors, entity_pair_negative_vectors], mode = l2_theo,  output_shape = ())
        return r_dot_e, r_dot_e_prime, reg
    else:
        return r_dot_e, r_dot_e_prime , 0


''' Energy based neural net models '''           

def getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb, opts): 
    vect_dim      = opts.vect_dim
    neg_samples   = opts.neg_samples
    num_entity_pairs  = opts.num_entity_pairs
    num_relations = opts.num_relations
    l2_reg_entity_pair = opts.l2_entity
    l2_reg_relation = opts.l2_relation    

    if opts.oov_train:
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs+1, init='normal', name = 'entity_embeddings')#,  W_regularizer=l2(l2_reg_entity_pair))
    else:
        entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs, init='normal', name = 'entity_embeddings')#,  W_regularizer=l2(l2_reg_entity_pair))

    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')#, W_regularizer=l2(l2_reg_relation))

    entity_pair_vectors = Flatten()(entity_pairs(kb_entity_pairs))
    entity_pair_negative_vectors = entity_pairs(neg_samples_kb)

    relation_vectors = Flatten()(relations(kb_relations))
    r_dot_e = merge([relation_vectors, entity_pair_vectors], mode = lambda X: T.batched_dot(X[0], X[1]), output_shape = ())
    r_dot_e_prime = merge([relation_vectors, entity_pair_negative_vectors], mode ='dot', output_shape = (neg_samples), dot_axes=(1,2))

    def l2_theo(input):
        entity_vectors, relation_vectors, entity_negative_vectors = input
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )
        return (T.sqr(entity_vectors).sum()/denom) + (T.sqr(entity_negative_vectors).sum()/denom) + (T.sqr(relation_vectors).mean())

    if opts.theo_reg:
        reg = merge([entity_pair_vectors, relation_vectors, entity_pair_negative_vectors], mode = l2_theo,  output_shape = ())
        return r_dot_e, r_dot_e_prime, entity_pairs, reg
    else:
        return r_dot_e, r_dot_e_prime, entity_pairs, 0

    return r_dot_e, r_dot_e_prime, entity_pairs

def get_cross(i, neg_samples):
    def cross_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1)*id_ei
    return lambda X: cross_fn(X[0], X[1])

def getDM_score_joint(kb_entities_1, kb_entities_2, kb_relations, neg_samples_kb, entities, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation  
    '''
        while reading some models are stores with entity embeddings named as entity_embeddings, and some as entity_embeddings_DM
    ''' 
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')#, W_regularizer=l2(l2_reg_relations))

    entity_vectors_e1 = Flatten()(entities(kb_entities_1))
    entity_vectors_e2 = Flatten()(entities(kb_entities_2))
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors = Flatten()(relations(kb_relations))

    get_cross_1 = get_cross(0, neg_samples)
    get_cross_2 = get_cross(1, neg_samples)
    e1_cross_e2_prime = merge([entity_vectors_e1, entity_negative_vectors], mode = get_cross_1, output_shape = (neg_samples, vect_dim))
    e1_prime_cross_e2 = merge([entity_vectors_e2, entity_negative_vectors], mode = get_cross_2, output_shape = (neg_samples, vect_dim))
    e1_cross_e2       = merge([entity_vectors_e1, entity_vectors_e2], mode=lambda X: X[0]*X[1], output_shape =(vect_dim,))
    
    score_DM = merge([relation_vectors, e1_cross_e2], mode = lambda X : T.batched_dot(X[0], X[1]), output_shape=(1,))
    score_DM_e2_corrupted = merge([relation_vectors, e1_cross_e2_prime], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
    score_DM_e1_corrupted = merge([relation_vectors, e1_prime_cross_e2], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

    return score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted

    
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
   
    reg = 0
    r_dot_e, r_dot_e_prime, reg = getMF_score(kb_entity_pairs, kb_relations, neg_samples_kb, opts) 
    if opts.loss == "ll":
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF, output_shape = (1,))
    elif opts.loss == "mm":
        print "Using Max-margin loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_max_margin, output_shape = (1,))
    elif opts.loss == "logistic":
        print "Using Logistic loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_logistic, output_shape = (1,))

    if reg:
        print "Model Regularization!", opts.theo_reg#opts.l2_entity_pair
        score_kb = Lambda(lambda x: x[0] + opts.theo_reg * x[1], output_shape = (1,))([score_kb, reg])
 
    if optimizer=='Adagrad':
        if opts.loss == "logistic":
            print "Using Logistic Loss :: Adagrad :: Clip norm 1 "
            alg = Adagrad(lr=opts.lr, clipnorm=1)
        else:
            alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)

    model = Model(input=[kb_entity_pairs, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    return model


def build_MFModel_eEPContraint(opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_relations = opts.num_relations
    warm_start    = opts.warm_start
    optimizer     = opts.optimizer
    num_entity_pairs  = opts.num_entity_pairs
    num_entities  = opts.num_entities
    l2_reg_entity_pair = opts.l2_entity_pair
    l2_reg_relation = opts.l2_relation    

    #define all inputs
    kb_entity_pairs = Input(shape=(1,), dtype='int32', name='kb_entity_pairs')
    kb_entities_1   = Input(shape=(1,), dtype='int32', name='kb_entities_1')
    kb_entities_2   = Input(shape=(1,), dtype='int32', name='kb_entities_2')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    neg_samples_kb  = Input(shape=(neg_samples, ), dtype = 'int32', name='kb_neg_examples')
   

    r_dot_e, r_dot_e_prime, entity_pairs, reg = getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb, opts) 
    if opts.loss == "ll":
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF, output_shape = (1,))
    elif opts.loss == "mm":
        print "Using Max-margin loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_max_margin, output_shape = (1,))
    elif opts.loss == "logistic":
        print "Logistic loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_logistic, output_shape = (1,))
        
    if optimizer=='Adagrad':
        if opts.loss == "logistic":
            alg = Adagrad(lr=opts.lr, clipnorm=1) #Theo's code use clipnorm=1
            print "Using Adagrad w/t max grad norm = 1"
        else:
            alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)

    if reg:
        print "Model Regularization!", opts.theo_reg
        score_kb = Lambda(lambda x: x[0] + opts.theo_reg * x[1], output_shape = (1,))([score_kb, reg])

    model = Model(input=[kb_entity_pairs, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    
    score_constraint, _ = getEP_score_joint(kb_entity_pairs, kb_entities_1, kb_entities_2, entity_pairs, opts)
    aux_model = Model(input=[kb_entity_pairs, kb_entities_1, kb_entities_2], output=score_constraint)
    aux_model.compile(loss = lossFn, optimizer=alg)
    
    #aux_model.compile(loss = 'mean_squared_error', optimizer=alg)
    return model, aux_model

def get_optimizer(opts):
    optimizer = opts.optimizer
    if optimizer=='Adagrad':
        if opts.loss == "logistic":
            alg = Adagrad(lr=opts.lr, clipnorm=1) #Theo's code use clipnorm=1
            print "Using Adagrad w/t max grad norm = 1"
        else:
            alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    elif optimizer=='Adam':
        alg = Adam(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)
    
    return alg

def build_MFModel_eEPContraint_DM(opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_relations = opts.num_relations
    warm_start    = opts.warm_start
    optimizer     = opts.optimizer
    num_entity_pairs  = opts.num_entity_pairs
    num_entities  = opts.num_entities
    l2_reg_entity_pair = opts.l2_entity_pair
    l2_reg_relation = opts.l2_relation    

    #define all inputs
    kb_entity_pairs = Input(shape=(1,), dtype='int32', name='kb_entity_pairs')
    kb_entities_1   = Input(shape=(1,), dtype='int32', name='kb_entities_1')
    kb_entities_2   = Input(shape=(1,), dtype='int32', name='kb_entities_2')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    kb_relations_DM = Input(shape=(1,), dtype='int32', name='kb_relations_DM')
    neg_samples_kb  = Input(shape=(neg_samples, ), dtype = 'int32', name='kb_neg_examples')
    neg_samples_kb_DM=Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples_DM')

    r_dot_e, r_dot_e_prime, entity_pairs, reg = getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb, opts) 
    if opts.loss == "ll":
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF, output_shape = (1,))
    elif opts.loss == "mm":
        print "Using Max-margin loss"
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_max_margin, output_shape = (1,))
    elif opts.loss == "logistic":
        score_kb  = merge([r_dot_e, r_dot_e_prime], mode = custom_fn_MF_logistic, output_shape = (1,))

        
    alg = get_optimizer(opts)

    model = Model(input=[kb_entity_pairs, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    
    ##Model 2: EPEconstraint
    score_constraint, entities = getEP_score_joint(kb_entity_pairs, kb_entities_1, kb_entities_2, entity_pairs, opts)
    aux_model = Model(input=[kb_entity_pairs, kb_entities_1, kb_entities_2], output=score_constraint)
    aux_model.compile(loss = lossFn, optimizer=alg)
    
    ##model 3: DM
    score, score_e1_corrupted, score_e2_corrupted = getDM_score_joint(kb_entities_1, kb_entities_2, kb_relations_DM, neg_samples_kb_DM, entities, opts)

    if opts.loss == "ll":
        print "using softmax loss"
        score_kb_DM  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_softmax_approx, output_shape = (1,))#Log likelihod loss
    elif opts.loss =="mm":
        print "using max margin loss"
        score_kb_DM  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_max_margin, output_shape = (1,))#Max margin loss
    elif opts.loss == "logistic":
        score_kb_DM  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_logistic, output_shape = (1,))

    aux_model_2 = Model(input=[kb_entities_1, kb_entities_2, kb_relations_DM, neg_samples_kb_DM], output=score_kb_DM)
    aux_model_2.compile(loss = lossFn, optimizer=alg)

    #aux_model.compile(loss = 'mean_squared_error', optimizer=alg)
    return model, aux_model, aux_model_2



