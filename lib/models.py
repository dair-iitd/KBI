from keras.layers import *
from lib.wrappers import *
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.layers.core import Lambda
from keras.models import Model
import theano
import theano.tensor as T
import numpy as np
import sys
from keras.layers import *
import theano.tensor as T
import theano
import numpy as np
    

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

def get_max_margin(input):
    score, score_e1_corrupt, score_e2_corrupt = input
    net_loss =  T.sum(T.maximum(0,1.0 + score_e1_corrupt - score.dimshuffle(0,'x')), axis=1)
    net_loss += T.sum(T.maximum(0,1.0 + score_e2_corrupt - score.dimshuffle(0, 'x')), axis=1) #margin = 1.0
    return 1*net_loss

def get_optimizer(opts):
    optimizer = opts.optimizer
    if optimizer=='Adagrad':
        alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    elif optimizer=='Adam':
        alg = Adam(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)
    
    return alg


def lossFn(y_true, y_pred):
    return T.mean(y_pred)


def get_cross(i, neg_samples):
    def cross_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:,i]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1)*id_ei
    return lambda X: cross_fn(X[0], X[1])

def prepare_concat(i, neg_samples, dim, entity_vecs, entity_negative_vecs, relation_vecs):
    id_ei = Lambda(lambda X: X[:, i*neg_samples : (i+1)*neg_samples], output_shape = (neg_samples, dim))(entity_negative_vecs)
    ei_tiled = Lambda(lambda X : T.tile(X[:,i].dimshuffle(0,'x',1), reps=(1,neg_samples,1)),output_shape = (neg_samples, dim))(entity_vecs)
    r_tiled  = Lambda(lambda X : T.tile(X.dimshuffle(0,'x',1), reps=(1,neg_samples,1)),output_shape = (neg_samples, dim))(relation_vecs)
    if i:
        return merge([ei_tiled, r_tiled, id_ei], mode = 'concat', output_shape = (neg_samples, dim*3))
    else:
        return merge([id_ei, r_tiled, ei_tiled], mode = 'concat', output_shape = (neg_samples, dim*3))



def concat_e1_r_e2(X_in):

    e1 = Lambda(lambda X : X_in[0][:,0], output_shape=(100,))(X_in)#.dimshuffle(0,1))(X)
    e2 = Lambda(lambda X : X_in[0][:,1], output_shape=(100,))(X_in)#.dimshuffle(0,1))(X)
    r  = Lambda(lambda X : X_in[1], output_shape=(100,))(X_in)
    concat = merge([e1, r, e2], mode='concat', concat_axis=-1)
    return concat

def cross_e1_e2(X):
    e1 = X[:,0]
    e2 = X[:,1]
    return e1*e2

def get_minus(i, neg_samples):
    def minus_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:,i]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1) - id_ei
    return lambda X: minus_fn(X[0], X[1])

def minus_e1_e2(X):
    e1 = X[:,0]
    e2 = X[:,1]
    return e1 - e2

def get_dot(i):
    def dot_fn(relation_vecs, entity_vecs):
        ei = entity_vecs[:,i]
        dotProd = T.batched_dot(relation_vecs, ei)
        return dotProd
    return lambda X: dot_fn(X[0], X[1])

def get_dot_neg(i, neg_samples):
    def dot_fn(relation_vecs, entity_negative_vecs):
        id_ei = entity_negative_vecs[:, i*neg_samples: (i+1)*neg_samples]
        return T.batched_dot(id_ei,relation_vecs)
    return lambda X: dot_fn(X[0], X[1]) 


def concat(i, neg_samples):
    def concat_fn(entity_vecs, entity_negative_vecs):
        ei = T.tile(entity_vecs[:,i].dimshuffle(0,'x',1), (1, neg_samples, 1))
        negVecs = entity_negative_vecs[:, i*neg_samples: (i+1)*neg_samples]
        if (i==0):
            # e1 concatenated with e2'
            return T.concatenate([ei, negVecs], axis=-1)
        else:
            # e1' concatenated with e1
            return T.concatenate([negVecs, ei], axis=-1)    
        
    return lambda X: concat_fn(X[0], X[1])


def concat_e1_e2(X):
    e1 = X[:,0]
    e2 = X[:,1]
    return T.concatenate([e1,e2], axis=-1)

def inner_prod(x1, x2, x3):
    return T.batched_dot(x1*x2, x3)

def hermitian_product(X):
    e_real, e_im, r_real, r_im = X
    
    e1_real = e_real[:, 0]; e2_real = e_real[:,1]
    e1_im = e_im[:, 0]; e2_im = e_im[:, 1] 
    
    return inner_prod(e1_real, e2_real, r_real) + inner_prod(e1_im, e2_im, r_real) + inner_prod(e1_real, e2_im, r_im) - inner_prod(e1_im, e2_real, r_im)



''' done '''
def getTransE_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation   
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings', W_regularizer=l2(l2_reg_entities))
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg_relations))
    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors = Flatten()(relations(kb_relations))

    get_minus_1 = get_minus(0, neg_samples)
    get_minus_2 = get_minus(1, neg_samples)
    e1_minus_e2_prime = merge([entity_vectors, entity_negative_vectors], mode = get_minus_1, output_shape = (neg_samples, vect_dim))#batch_size x neg_samples x vect_dim
    e1_prime_minus_e2 = merge([entity_vectors, entity_negative_vectors], mode = get_minus_2, output_shape = (neg_samples, vect_dim))
    e1_minus_e2    = Lambda(minus_e1_e2, output_shape = (vect_dim,))(entity_vectors)#batch_size x vect_dim
    
    score_TransE = merge([e1_minus_e2, relation_vectors], mode = lambda X: -1.0*T.sqrt(T.sum(T.sqr(X[0] + X[1]), axis = 1)), output_shape = ())                       
    score_TransE_e2_corrupted = merge([e1_minus_e2_prime, relation_vectors], mode = lambda X: -1.0*T.sqrt(T.sum(T.sqr(X[0] + X[1].dimshuffle(0, 'x', 1)), axis = 2)), output_shape = (neg_samples,))
    score_TransE_e1_corrupted = merge([e1_prime_minus_e2, relation_vectors], mode = lambda X: -1.0*T.sqrt(T.sum(T.sqr(X[0] + X[1].dimshuffle(0, 'x', 1)), axis = 2)), output_shape = (neg_samples,))

    return score_TransE, score_TransE_e1_corrupted, score_TransE_e2_corrupted

''' 
    Implementaiton of DistMult with Complex Embeddings from Trouillon et. al. (2016)
'''
def getComplex_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation  
    
    #inner_prod(e1_real, e2_real, r_real) + inner_prod(e1_im, e2_im, r_real) + inner_prod(e1_real, e2_im, r_im) - inner_prod(e1_im, e2_real, r_im)
    def accumulator(x, y, relation_vectors, f):
        a00 = merge([x[0], y[0]], mode = f, output_shape = (neg_samples, vect_dim)) 
        a01 = merge([x[0], y[1]], mode = f, output_shape = (neg_samples, vect_dim))
        a11 = merge([x[1], y[1]], mode = f, output_shape = (neg_samples, vect_dim))
        a10 = merge([x[1], y[0]], mode = f, output_shape = (neg_samples, vect_dim))

        r1 = merge([relation_vectors[0], a00], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r2 = merge([relation_vectors[0], a11], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r3 = merge([relation_vectors[1], a01], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r4 = merge([relation_vectors[1], a10], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

        result = merge([r1,r2,r3,r4], mode = lambda X : X[0]+X[1]+X[2]-X[3], output_shape=(neg_samples,))
        return result


    entities_real  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings_real', W_regularizer=l2(l2_reg_entities))
    entities_im  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings_im', W_regularizer=l2(l2_reg_entities))
    relations_real = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_real', W_regularizer=l2(l2_reg_relations))
    relations_im = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_im', W_regularizer=l2(l2_reg_relations))

    entity_vectors_real = entities_real(kb_entities)
    entity_vectors_im =  entities_im(kb_entities)

    entity_negative_vectors_real = entities_real(neg_samples_kb)
    entity_negative_vectors_im = entities_im(neg_samples_kb)

    relation_vectors_real = Flatten()(relations_real(kb_relations))   
    relation_vectors_im = Flatten()(relations_im(kb_relations))  


    get_cross_1 = get_cross(0, neg_samples)
    get_cross_2 = get_cross(1, neg_samples)


    score_complex = merge([entity_vectors_real, entity_vectors_im, relation_vectors_real, relation_vectors_im], mode = hermitian_product,  output_shape = ()) 


    score_complex_e1_corrupted = accumulator([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im],
                                             [relation_vectors_real, relation_vectors_im] ,get_cross_2)  

    score_complex_e2_corrupted = accumulator([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im], 
                                            [relation_vectors_real, relation_vectors_im] ,get_cross_1)  
        
    

    return score_complex, score_complex_e1_corrupted, score_complex_e2_corrupted


''' Done'''
def getDM_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation  
    '''
        while reading some models are stores with entity embeddings named as entity_embeddings, and some as entity_embeddings_DM
    ''' 
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings', W_regularizer=l2(l2_reg_entities))
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg_relations))

    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors = Flatten()(relations(kb_relations))

    get_cross_1 = get_cross(0, neg_samples)
    get_cross_2 = get_cross(1, neg_samples)
    e1_cross_e2_prime = merge([entity_vectors, entity_negative_vectors], mode = get_cross_1, output_shape = (neg_samples, vect_dim))
    e1_prime_cross_e2 = merge([entity_vectors, entity_negative_vectors], mode = get_cross_2, output_shape = (neg_samples, vect_dim))
    e1_cross_e2    = Lambda(cross_e1_e2, output_shape = (vect_dim,))(entity_vectors)

    score_DM = merge([relation_vectors, e1_cross_e2], mode = lambda X : T.batched_dot(X[0], X[1]), output_shape=())
    score_DM_e2_corrupted = merge([relation_vectors, e1_cross_e2_prime], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
    score_DM_e1_corrupted = merge([relation_vectors, e1_prime_cross_e2], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

    return score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted


def get_forward_pass(layers):   
    def run(input):
        output = layers[0](input)

        if len(layers)>1:
            for i in xrange(1, len(layers)):
                output = layers[i](output)

        return output

    return run


def getConcat_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity
    l2_reg_relations = opts.l2_relation
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings_DM', W_regularizer=l2(l2_reg_entities))
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg_relations))

    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors = Flatten()(relations(kb_relations))

    e1_concat_e2_prime_concat_r = prepare_concat(0, neg_samples, vect_dim , entity_vectors, entity_negative_vectors, relation_vectors)
    e1_prime_concat_e2_concat_r = prepare_concat(1, neg_samples, vect_dim , entity_vectors, entity_negative_vectors, relation_vectors)

    e1_concat_e2_concat_r       = merge([entity_vectors, relation_vectors], mode=concat_e1_r_e2, output_shape = (vect_dim*3,))
    
    #define the neural network for outputing new corrupted and real scores
    nnet_layer_1 = Dense(100 ,init='glorot_normal',activation='tanh', name='dense_1')
    nnet_layer_2 = Dense(10 ,init='glorot_normal',activation='tanh', name='dense_2')
    nnet_layer_3 = Dense(1  ,init='glorot_normal',activation='sigmoid', name='dense_3')

    layers = [nnet_layer_1, nnet_layer_2, nnet_layer_3]
    # create a function to perform forward passes over the neural network
    forward_pass = get_forward_pass(layers)
    forward_pass_distributed = get_forward_pass([TimeDistributed(layer) for layer in layers])

    score_concat = forward_pass(e1_concat_e2_concat_r)
    score_concat_e1_corrupted = Flatten()(forward_pass_distributed(e1_prime_concat_e2_concat_r))
    score_concat_e2_corrupted = Flatten()(forward_pass_distributed(e1_concat_e2_prime_concat_r))

    score_concat = Lambda(lambda X: X[:,0])(score_concat)

    return score_concat, score_concat_e1_corrupted, score_concat_e2_corrupted


''' Done '''
def getE_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation

    entities    = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings',  W_regularizer=l2(l2_reg_entities))
    relations_s = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_s', W_regularizer=l2(l2_reg_relations))
    relations_o = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_o', W_regularizer=l2(l2_reg_relations))


    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors_s = Flatten()(relations_s(kb_relations))
    relation_vectors_o = Flatten()(relations_o(kb_relations))

    get_dot_1 =  get_dot(0)
    get_dot_2 =  get_dot(1)
    get_dot_neg_1 = get_dot_neg(0, neg_samples)
    get_dot_neg_2 = get_dot_neg(1, neg_samples)
    rs_dot_e1 = merge([relation_vectors_s, entity_vectors], mode =get_dot_1, output_shape = ())
    ro_dot_e2 = merge([relation_vectors_o, entity_vectors], mode =get_dot_2, output_shape = ())
    rs_dot_e1_prime = merge([relation_vectors_s, entity_negative_vectors], mode =get_dot_neg_2, output_shape = (neg_samples,))
    ro_dot_e2_prime = merge([relation_vectors_o, entity_negative_vectors], mode =get_dot_neg_1, output_shape = (neg_samples,))

    score_E = merge([rs_dot_e1, ro_dot_e2], mode = lambda X : X[0]+X[1], output_shape=())

    score_E_e2_corrupted = merge([rs_dot_e1, ro_dot_e2_prime], mode = lambda X: X[0].dimshuffle(0,'x') + X[1], output_shape=(neg_samples,))
    score_E_e1_corrupted = merge([ro_dot_e2, rs_dot_e1_prime], mode = lambda X: X[0].dimshuffle(0,'x') + X[1], output_shape=(neg_samples,))

    return score_E, score_E_e1_corrupted, score_E_e2_corrupted


''' Energy based neural net models '''     


def build_biLSTM(opts):
    vect_dim = opts.vect_dim
    num_entities  = opts.num_entities
    relations_vocab = opts.relation_vocab
    lstm_units    = opts.lstm_units
    sent_size = opts.sent_size
    neg_samples = opts.neg_samples
    l2_reg  = opts.l2_entity

    ''' Bi-LSTM model for capturing compositionality '''

    kb_relations    = Input(shape=(sent_size,), dtype='int32', name='kb_relations')
    # +1 for the padding
    relations = Embedding(output_dim=vect_dim, input_dim=relations_vocab+1, input_length=sent_size,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg))

    relation_vectors = relations(kb_relations)

    forward  = LSTM(lstm_units, return_sequences=False, name="forward")(relation_vectors)
    backward = LSTM(lstm_units, return_sequences=False, go_backwards=True, name="backward")(relation_vectors)
    forward_backward = merge([forward, backward], mode="concat", concat_axis=1, name="forward_backward")
    dropout = Dropout(0.5, name = "dropout_fb")(forward_backward)
    sentence_vectors_1 = Dense(vect_dim, W_regularizer=l2(l2_reg), activation= 'relu')(dropout)
    sentence_vectors = Dense(vect_dim, W_regularizer=l2(l2_reg), activation= 'relu')(sentence_vectors_1)

    aux_model = Model([kb_relations], output = sentence_vectors)
    aux_model.compile(loss='mean_squared_error', optimizer='sgd')


    ''' Once relation embedding is obtained, use standard distMult for calculating scores '''   
    kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
    neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')


    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities, init='normal',name = 'entity_embeddings', W_regularizer=l2(l2_reg))
    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)


    get_cross_1 = get_cross(0, neg_samples)
    get_cross_2 = get_cross(1, neg_samples)
    e1_cross_e2_prime = merge([entity_vectors, entity_negative_vectors], mode = get_cross_1, output_shape = (None, neg_samples, vect_dim))
    e1_prime_cross_e2 = merge([entity_vectors, entity_negative_vectors], mode = get_cross_2, output_shape = (None, neg_samples, vect_dim))

    e1_cross_e2 = Lambda(cross_e1_e2, output_shape = (vect_dim,))(entity_vectors)

    score_kb = merge([e1_cross_e2, e1_cross_e2_prime, e1_prime_cross_e2, sentence_vectors], mode=custom_fn, output_shape = (None, 1))   
    alg = get_optimizer(opts)
    model = Model(input = [kb_entities, kb_relations, neg_samples_kb], output = score_kb) 
    model.compile(loss=lossFn, optimizer=alg)

    return model, aux_model



def build_atomic_model(opts, model_func):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    optimizer      = opts.optimizer
    l2_reg = opts.l2_entity    
    #define all inputs
    kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')

    #define embeddings
     
    score, score_e1_corrupted, score_e2_corrupted  = model_func(kb_entities, kb_relations, neg_samples_kb, opts)

    if opts.loss == "ll":
        print "using softmax loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_softmax_approx, output_shape = (1,))#Log likelihod loss
    elif opts.loss =="mm":
        print "using max margin loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_max_margin, output_shape = (1,))#Max margin loss

    alg = get_optimizer(opts)

    model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    

    return model



