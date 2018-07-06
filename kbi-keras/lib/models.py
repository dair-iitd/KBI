from keras.layers import *
from lib.wrappers import *
from lib.tensor_manipulators import *
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
    

# === Implemention of energies of various Tensor Factorization models (E, DistMult, Complex DistMult, TransE)

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


def getComplex_score(kb_entities, kb_relations, neg_samples_kb, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    l2_reg_relations = opts.l2_relation  
    
    #inner_prod(e1_real, e2_real, r_real) + inner_prod(e1_im, e2_im, r_real) + inner_prod(e1_real, e2_im, r_im) - inner_prod(e1_im, e2_real, r_im)
    def hermitian_product_negatives(x, y, relation_vectors, f):
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
    score_complex_e1_corrupted = hermitian_product_negatives([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im],
                                             [relation_vectors_real, relation_vectors_im] ,get_cross_2)  

    score_complex_e2_corrupted = hermitian_product_negatives([entity_vectors_real, entity_vectors_im], [entity_negative_vectors_real, entity_negative_vectors_im], 
                                            [relation_vectors_real, relation_vectors_im] ,get_cross_1)  
        
    
    return score_complex, score_complex_e1_corrupted, score_complex_e2_corrupted


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





# == A generic function for constructing Tensor Factorization models. Requires a model_func that returns energies of positive and negative examples. Returns a loss
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
        score_kb_1  = merge([score, score_e1_corrupted], mode = softmax_approx, output_shape = (1,))
        score_kb_2 = merge([score, score_e2_corrupted], mode = softmax_approx, output_shape = (1,))
    elif opts.loss =="mm":
        print "using max margin loss"
        score_kb_1  = merge([score, score_e1_corrupted], mode = get_max_margin, output_shape = (1,))
        score_kb_2 = merge([score, score_e2_corrupted], mode = max_margin, output_shape = (1,))


    score_kb = merge([score_kb_1 , score_kb_2], lambda X: X[0] + X[1], output_shape = (1,))

    alg = get_optimizer(opts)

    model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
    model.compile(loss = lossFn, optimizer=alg)
    

    return model



