import joblib
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
import keras,random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,SimpleRNN
from keras import backend as K
from keras import initializers 

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

def get_logistic(input):
    score, score_e1_corrupt, score_e2_corrupt = input
       
    #if random.random() > 0.5: 
    #    loss_e_corrupt = T.sum(T.nnet.softplus(1*score_e1_corrupt), axis=1)
    #else:
    #    loss_e_corrupt = T.sum(T.nnet.softplus(1*score_e2_corrupt), axis=1)

    loss_e1_corrupt = T.sum(T.nnet.softplus(1*score_e1_corrupt), axis=1)
    loss_e2_corrupt = T.sum(T.nnet.softplus(1*score_e2_corrupt), axis=1)
    loss_score = T.nnet.softplus(-1*score)

    net_loss = loss_e1_corrupt + loss_e2_corrupt + loss_score

    net_loss = net_loss/(1+2*score_e1_corrupt.shape[1])#T.exp(T.log(net_loss)-T.log((1+2*score_e1_corrupt.shape[1])))

    return 1*net_loss

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


def lossFn(y_true, y_pred):
    return T.mean(y_pred)


def get_cross(i, neg_samples):
    def cross_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:,i]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1)*id_ei
    return lambda X: cross_fn(X[0], X[1])

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
#each batch consists of independent kb samples and multiple type training pairs
def get_type_scores(kb_entities, kb_relations, neg_samples_kb, positive_pairs, negative_pairs, opts):
    neg_samples   = opts.neg_samples
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity
    l2_reg_relations = opts.l2_relation
    type_pair_count = opts.type_pair_count
    type_dim = opts.type_dim
    entity_type = Embedding(output_dim=type_dim, input_dim=num_entities+1, init='normal', name='entity_type_embeddings', embeddings_regularizer=l2(l2_reg_entities))
    relation_head_type = Embedding(output_dim=type_dim, input_dim=num_relations, init='normal', name='relation_head_type_embeddings', embeddings_regularizer=l2(l2_reg_relations))
    relation_tail_type = Embedding(output_dim=type_dim, input_dim=num_relations, init='normal', name='relation_tail_type_embeddings', embeddings_regularizer=l2(l2_reg_relations))

    entity_type_vectors = entity_type(kb_entities)
    entity_negative_type_vectors = entity_type(neg_samples_kb)
    relation_head_type_vectors = Flatten()(relation_head_type(kb_relations))
    relation_tail_type_vectors = Flatten()(relation_tail_type(kb_relations))
    if(positive_pairs is not None):
        positive_type_vectors = entity_type(positive_pairs)
    else:
        print("no +ve pairs")
        positive_type_vectors = None
    if(negative_pairs is not None):
        negative_type_vectors = entity_type(negative_pairs)
    else:
        print("No -ve pairs")
        negative_type_vectors = None
    get_dot_1 = get_dot(0)
    get_dot_2 = get_dot(1)
    get_dot_neg_1 = get_dot_neg(0, neg_samples)
    get_dot_neg_2 = get_dot_neg(1, neg_samples)
    rht_dot_e1 = merge([relation_head_type_vectors, entity_type_vectors], mode=get_dot_1,output_shape=())
    rtt_dot_e2 = merge([relation_tail_type_vectors, entity_type_vectors], mode=get_dot_2,output_shape=())
    rht_dot_e1_prime = merge([relation_head_type_vectors, entity_negative_type_vectors], mode=get_dot_neg_1, output_shape=(neg_samples,))
    rht_dot_e2_prime = merge([relation_tail_type_vectors, entity_negative_type_vectors], mode=get_dot_neg_2, output_shape=(neg_samples,))
    rht_dot_e1 = Activation('sigmoid')(rht_dot_e1)
    rtt_dot_e2 = Activation('sigmoid')(rtt_dot_e2)
    rht_dot_e1_prime = Activation('sigmoid')(rht_dot_e1_prime)
    rht_dot_e2_prime = Activation('sigmoid')(rht_dot_e2_prime)
    score_type_term = merge([rht_dot_e1, rtt_dot_e2], mode=lambda x:x[0]*x[1], output_shape=())
    score_type_e2_corrupted = merge([rht_dot_e1, rht_dot_e2_prime], mode=lambda x:x[0].dimshuffle(0, 'x')*x[1], output_shape=(neg_samples,))
    score_type_e1_corrupted = merge([rtt_dot_e2, rht_dot_e1_prime], mode=lambda x:x[0].dimshuffle(0, 'x')*x[1], output_shape=(neg_samples,))
    if(positive_type_vectors is not None):
        positive_pair_dot = Lambda(lambda x:T.sum(x[:, :, 0, :]*x[:, :, 1, :], axis=2), output_shape=(type_pair_count,))(positive_type_vectors)
    else:
        print("ppd none");
        positive_pair_dot = None
    if(negative_type_vectors is not None):
        negative_pair_dot = Lambda(lambda x:T.sum(x[:, :, 0, :]*x[:, :, 1, :], axis=2), output_shape=(type_pair_count,))(negative_type_vectors)
    else:
        print("npd none")
        negative_pair_dot = None

    return score_type_term, score_type_e1_corrupted, score_type_e2_corrupted, positive_pair_dot, negative_pair_dot


def getTransE_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation   
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings')#, W_regularizer=l2(l2_reg_entities))
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')#, W_regularizer=l2(l2_reg_relations))

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

    def l2_theo(input):
        entity_vectors, relation_vectors, entity_negative_vectors = input
        all_e1_real = entity_vectors[:, 0]
        all_e2_real = entity_vectors[:, 1]
        all_ne1_real = entity_negative_vectors[:, 0*neg_samples : (0+1)*neg_samples]
        all_ne2_real = entity_negative_vectors[:, 1*neg_samples : (1+1)*neg_samples]
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )

        return (T.sqr(all_e1_real).sum()/denom) + (T.sqr(all_e2_real).sum()/denom) + (T.sqr(relation_vectors).mean()) + (T.sqr(all_ne1_real).sum()/denom) + (T.sqr(all_ne2_real).sum()/denom)

    if opts.theo_reg:
        reg = merge([entity_vectors, relation_vectors, entity_negative_vectors], mode = l2_theo,  output_shape = ())
        return score_TransE, score_TransE_e1_corrupted, score_TransE_e2_corrupted, reg
    else:
        return score_TransE, score_TransE_e1_corrupted, score_TransE_e2_corrupted, 0


''' 
    Implementaiton of DistMult with Complex Embeddings from Trouillon et. al. (2016)
'''
def getComplex_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    #1
    l2_reg_relations = opts.l2_relation  #1
    
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

    def accumulator_e2(x, y, relation_vectors, f):
        a00 = merge([x[0], y[0]], mode = f, output_shape = (neg_samples, vect_dim)) 
        a01 = merge([x[0], y[1]], mode = f, output_shape = (neg_samples, vect_dim))
        a11 = merge([x[1], y[1]], mode = f, output_shape = (neg_samples, vect_dim))
        a10 = merge([x[1], y[0]], mode = f, output_shape = (neg_samples, vect_dim))

        r1 = merge([relation_vectors[0], a00], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r2 = merge([relation_vectors[0], a11], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r3 = merge([relation_vectors[1], a01], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))
        r4 = merge([relation_vectors[1], a10], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

        result = merge([r1,r2,r3,r4], mode = lambda X : X[0]+X[1]-X[2]+X[3], output_shape=(neg_samples,))
        return result

    mu = 0;sigma = 0.05#1#0.5#1#0.05
    print mu,sigma
    path = "/home/cse/phd/csz148211/code/joint_embedding/code_valid/tmp-test/"
    randn_init_embed = np.random.normal(mu, sigma, (num_entities+1,vect_dim))#np.random.randn(num_entities+1,vect_dim)
    #randn_init_embed = joblib.load(path+"e1_embedding_init.joblib")
    entities_real  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, weights=[randn_init_embed],name = 'entity_embeddings_real')#, W_regularizer=l2(l2_reg_entities))
    randn_init_embed = np.random.normal(mu, sigma, (num_entities+1,vect_dim))#randn_init_embed = np.random.randn(num_entities+1,vect_dim)
    #randn_init_embed = joblib.load(path+"e2_embedding_init.joblib")
    entities_im  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, weights=[randn_init_embed],name = 'entity_embeddings_im')#, W_regularizer=l2(l2_reg_entities))
    randn_init_embed = np.random.normal(mu, sigma, (num_relations,vect_dim))#randn_init_embed = np.random.randn(num_relations,vect_dim)
    #randn_init_embed = joblib.load(path+"r1_embedding_init.joblib")
    relations_real = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1, weights=[randn_init_embed], name='relation_embeddings_real')#, W_regularizer=l2(l2_reg_relations))
    randn_init_embed = np.random.normal(mu, sigma, (num_relations,vect_dim))#randn_init_embed = np.random.randn(num_relations,vect_dim)
    #randn_init_embed = joblib.load(path+"r2_embedding_init.joblib")
    relations_im = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1, weights=[randn_init_embed], name='relation_embeddings_im')#, W_regularizer=l2(l2_reg_relations))
    
    entity_vectors_real = entities_real(kb_entities)
    entity_vectors_im =  entities_im(kb_entities)

    entity_negative_vectors_real = entities_real(neg_samples_kb)
    entity_negative_vectors_im = entities_im(neg_samples_kb)

    relation_vectors_real = Flatten()(relations_real(kb_relations))   
    relation_vectors_im = Flatten()(relations_im(kb_relations))  

    def l2_theo(input):
        entity_vectors_real, entity_vectors_im, relation_vectors_real, relation_vectors_im, entity_negative_vectors_real, entity_negative_vectors_im = input
        #return (T.sqr(entity_vectors_real).mean()) + (T.sqr(entity_vectors_im).mean()) + (T.sqr(relation_vectors_real).mean()) + (T.sqr(relation_vectors_im).mean()) + T.sqr(entity_negative_vectors_real).mean() + T.sqr(entity_negative_vectors_im).mean()  
        all_e1_real = entity_vectors_real[:, 0]
        all_e1_im   = entity_vectors_im[:, 0]
        all_e2_real = entity_vectors_real[:, 1]
        all_e2_im   = entity_vectors_im[:, 1]
        all_ne1_real = entity_negative_vectors_real[:, 0*neg_samples : (0+1)*neg_samples]
        all_ne1_im   = entity_negative_vectors_im[:, 0*neg_samples : (0+1)*neg_samples]
        all_ne2_real = entity_negative_vectors_real[:, 1*neg_samples : (1+1)*neg_samples]
        all_ne2_im   = entity_negative_vectors_im[:, 1*neg_samples : (1+1)*neg_samples]
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )
        
        return (T.sqr(all_e1_real).sum()/denom) + (T.sqr(all_e1_im).sum()/denom) + (T.sqr(all_e2_real).sum()/denom) + (T.sqr(all_e2_im).sum()/denom) + (T.sqr(relation_vectors_real).mean()) + (T.sqr(relation_vectors_im).mean()) + (T.sqr(all_ne1_real).sum()/denom) + (T.sqr(all_ne1_im).sum()/denom) + (T.sqr(all_ne2_real).sum()/denom) + (T.sqr(all_ne2_im).sum()/denom) 
        #return (T.sqr(entity_vectors_real).sum()/denom) + (T.sqr(entity_vectors_im).sum()/denom) + (T.sqr(relation_vectors_real).mean()) + (T.sqr(relation_vectors_im).mean()) + T.sqr(entity_negative_vectors_real).sum()/denom + T.sqr(entity_negative_vectors_im).sum()/denom 

    reg = merge([entity_vectors_real, entity_vectors_im, relation_vectors_real, relation_vectors_im, entity_negative_vectors_real, entity_negative_vectors_im], mode = l2_theo,  output_shape = ())

    get_cross_1 = get_cross(0, neg_samples)
    get_cross_2 = get_cross(1, neg_samples)
    
    score_complex = merge([entity_vectors_real, entity_vectors_im, relation_vectors_real, relation_vectors_im], mode = hermitian_product,  output_shape = ()) 

    score_complex_e1_corrupted = accumulator_e2([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im],
                                             [relation_vectors_real, relation_vectors_im] ,get_cross_2)  

    score_complex_e2_corrupted = accumulator([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im], 
                                            [relation_vectors_real, relation_vectors_im] ,get_cross_1)  
        
    return score_complex, score_complex_e1_corrupted, score_complex_e2_corrupted, reg


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
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings_DM')#, W_regularizer=l2(l2_reg_entities))
    relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')#, W_regularizer=l2(l2_reg_relations))

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

    def l2_theo(input):
        entity_vectors, relation_vectors, entity_negative_vectors = input
        #return (T.sqr(entity_vectors_real).mean()) + (T.sqr(entity_vectors_im).mean()) + (T.sqr(relation_vectors_real).mean()) + (T.sqr(relation_vectors_im).mean()) + T.sqr(entity_negative_vectors_real).mean() + T.sqr(entity_negative_vectors_im).mean()  
        all_e1_real = entity_vectors[:, 0]
        all_e2_real = entity_vectors[:, 1]
        all_ne1_real = entity_negative_vectors[:, 0*neg_samples : (0+1)*neg_samples]
        all_ne2_real = entity_negative_vectors[:, 1*neg_samples : (1+1)*neg_samples]
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )

        return (T.sqr(all_e1_real).sum()/denom) + (T.sqr(all_e2_real).sum()/denom) + (T.sqr(relation_vectors).mean()) + (T.sqr(all_ne1_real).sum()/denom) + (T.sqr(all_ne2_real).sum()/denom) 

    if opts.theo_reg:
        reg = merge([entity_vectors, relation_vectors, entity_negative_vectors], mode = l2_theo,  output_shape = ())

        return score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted, reg
    else:
        return score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted , 0


def get_forward_pass(layers):   
    def run(input):
        output = layers[0](input)

        if len(layers)>1:
            for i in xrange(1, len(layers)):
                output = layers[i](output)

        return output

    return run


''' Done '''
def getE_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation

    entities    = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings')#,  W_regularizer=l2(l2_reg_entities))
    relations_s = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_s')#, W_regularizer=l2(l2_reg_relations))
    relations_o = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_o')#, W_regularizer=l2(l2_reg_relations))


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

    def l2_theo(input):
        entity_vectors, relation_vectors_s, relation_vectors_o, entity_negative_vectors = input
        all_e1_real = entity_vectors[:, 0]
        all_e2_real = entity_vectors[:, 1]
        all_ne1_real = entity_negative_vectors[:, 0*neg_samples : (0+1)*neg_samples]
        all_ne2_real = entity_negative_vectors[:, 1*neg_samples : (1+1)*neg_samples]
        denom = opts.batch_size * opts.vect_dim * (1 +neg_samples )

        return (T.sqr(all_e1_real).sum()/denom) + (T.sqr(all_e2_real).sum()/denom) + (T.sqr(relation_vectors_o).mean()) + (T.sqr(all_ne1_real).sum()/denom) + (T.sqr(all_ne2_real).sum()/denom) + (T.sqr(relation_vectors_s).mean())

    if opts.theo_reg:
        reg = merge([entity_vectors, relation_vectors_s, relation_vectors_o, entity_negative_vectors], mode = l2_theo,  output_shape = ())
        return score_E, score_E_e1_corrupted, score_E_e2_corrupted, reg
    else:
        return score_E, score_E_e1_corrupted, score_E_e2_corrupted, 0


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

    reg = 0    
    score, score_e1_corrupted, score_e2_corrupted, reg  = model_func(kb_entities, kb_relations, neg_samples_kb, opts)

    if opts.loss == "ll":
        print "using softmax loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_softmax_approx, output_shape = (1,))#Log likelihod loss
    elif opts.loss =="mm":
        print "using max margin loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_max_margin, output_shape = (1,))#Max margin loss
    elif opts.loss == "logistic":
        print "using neg log-likelihood of logistic loss (Complex Embedding loss)"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_logistic, output_shape = (1,))
    
    if reg:
        print "Model Regularization!", opts.theo_reg
        score_kb = Lambda(lambda x: x[0] + opts.theo_reg * x[1], output_shape = (1,))([score_kb, reg])   

    alg = get_optimizer(opts)
    
    #score_kb = Lambda(lambda x: x[0] *1, output_shape = (1,))([reg])#score_e2_corrupted])

    model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    
    return model


def build_typed_model(opts, model_func):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    optimizer      = opts.optimizer
    type_pair_count = opts.type_pair_count
    l2_reg = opts.l2_entity
    type_dim = opts.type_dim
    #define all inputs
    kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')
    print("type pair count", type_pair_count)
    if(type_pair_count>0):
        positive_pairs  = Input(shape=(type_pair_count, 2,), dtype='int32', name='positive_pairs')
        negative_pairs  = Input(shape=(type_pair_count, 2,), dtype='int32', name='negative_pairs')
    else:
        positive_pairs, negative_pairs = None, None


    score_model, score_model_e1_corrupted, score_model_e2_corrupted, reg = model_func(kb_entities, kb_relations, neg_samples_kb, opts)
    score_model = Activation('sigmoid')(score_model)
    score_model_e1_corrupted = Activation('sigmoid')(score_model_e1_corrupted)
    score_model_e2_corrupted = Activation('sigmoid')(score_model_e2_corrupted)
    score_type, score_type_e1_corrupted, score_type_e2_corrupted, positive_pair_dot, negative_pair_dot = get_type_scores(kb_entities, kb_relations, neg_samples_kb, positive_pairs, negative_pairs, opts)

    factor = 20

    score = merge([score_model, score_type], lambda x:factor*x[0]*x[1], output_shape=(1,))
    score_e1_corrupted = merge([score_model_e1_corrupted, score_type_e1_corrupted], lambda x:factor*x[0]*x[1], output_shape=(neg_samples,))

    score_e2_corrupted = merge([score_model_e2_corrupted, score_type_e2_corrupted], lambda x:factor*x[0]*x[1], output_shape=(neg_samples,))

    if opts.loss == "ll":
        print "using softmax loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_softmax_approx, output_shape = (1,))#Log likelihod loss
    elif opts.loss =="mm":
        print "using max margin loss"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_max_margin, output_shape = (1,))#Max margin loss
    elif opts.loss == "logistic":
        print "using neg log-likelihood of logistic loss (Complex Embedding loss)"
        score_kb  = merge([score, score_e1_corrupted, score_e2_corrupted], mode = get_logistic, output_shape = (1,))

    alg = get_optimizer(opts)

    if reg:
        print "Model Regularization!", opts.theo_reg
        score_kb = Lambda(lambda x: x[0] + opts.theo_reg * x[1], output_shape = (1,))([score_kb, reg])

    if(type_pair_count > 0):
        type_scores = merge([positive_pair_dot, negative_pair_dot], mode=lambda x:1/(1+T.exp(T.mean(x[1])-T.mean(x[0]))), output_shape=(1,))
        final_score = merge([score_kb, type_scores], mode=lambda x: x[0]+x[1], output_shape=(1,))
        model = Model([kb_entities, kb_relations, neg_samples_kb, positive_pairs, negative_pairs], output=final_score)
        model.compile(loss=lossFn, optimizer=alg)
    else:
        model = Model([kb_entities, kb_relations, neg_samples_kb], output=score_kb)
        model.compile(loss=lossFn, optimizer=alg)
    return model

