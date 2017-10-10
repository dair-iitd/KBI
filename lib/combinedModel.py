from keras.layers import *
from lib.wrappers import *
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adagrad
from keras.constraints import maxnorm
from keras.layers.core import Lambda
from keras.models import Model
import theano
import theano.tensor as T
import numpy as np
import sys
from keras import backend as K
from lib.tensor_manipulators import *


def get_forward_pass(layers):   
    def run(input):
        output = layers[0](input)

        if len(layers)>1:
            for i in xrange(1, len(layers)):
                output = layers[i](output)

        return output

    return run




# == Energies for a Bunch of hybrid MF + TF models from our paper (refer https://arxiv.org/abs/1706.00637)

def getE_score_joint(kb_entities, kb_relations, neg_samples_kb, opts):
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
    rs_dot_e1 = merge([relation_vectors_s, entity_vectors], mode =get_dot_1, output_shape = ())
    ro_dot_e2 = merge([relation_vectors_o, entity_vectors], mode =get_dot_2, output_shape = ())
    ro_dot_e2_prime = merge([relation_vectors_o, entity_negative_vectors], mode =get_dot_neg_1, output_shape = (neg_samples,))

    score_E = merge([rs_dot_e1, ro_dot_e2], mode = lambda X : X[0]+X[1], output_shape=())

    score_E_e2_corrupted = merge([rs_dot_e1, ro_dot_e2_prime], mode = lambda X: X[0].dimshuffle(0,'x') + X[1], output_shape=(neg_samples,))

    if opts.add_loss:
        get_dot_neg_2 = get_dot_neg(1, neg_samples)
        rs_dot_e1_prime = merge([relation_vectors_s, entity_negative_vectors], mode =get_dot_neg_2, output_shape = (neg_samples,))
        score_E_e1_corrupted = merge([ro_dot_e2, rs_dot_e1_prime], mode = lambda X: X[0].dimshuffle(0,'x') + X[1], output_shape=(neg_samples,))

    else:
        score_E_e1_corrupted = None
    
    return score_E, score_E_e1_corrupted, score_E_e2_corrupted


def getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb, relations, opts): 
    vect_dim      = opts.vect_dim
    neg_samples   = opts.neg_samples
    num_entity_pairs  = opts.num_entity_pairs
    num_relations = opts.num_relations
    l2_reg_entity_pair = opts.l2_entity_pair
    print("regularizing MF entity embeddings with l2 penalty of: %5.4f" %l2_reg_entity_pair)
    # +1 for the OOV embedding. This might change to support OOV embeddings of the form (e1, ?) instead.
    entity_pairs  = Embedding(output_dim=vect_dim, input_dim=num_entity_pairs+1, init='normal', name = 'entity_embeddings_MF',  W_regularizer=l2(l2_reg_entity_pair))

    entity_pair_vectors = Flatten()(entity_pairs(kb_entity_pairs))
    entity_pair_negative_vectors = entity_pairs(neg_samples_kb)

    relation_vectors = Flatten()(relations(kb_relations))
    r_dot_e = merge([relation_vectors, entity_pair_vectors], mode = lambda X: T.batched_dot(X[0], X[1]), output_shape = ())
    r_dot_e_prime = merge([relation_vectors, entity_pair_negative_vectors], mode ='dot', output_shape = (neg_samples), dot_axes=(1,2))


    return r_dot_e, r_dot_e_prime


def getDM_score_joint(kb_entities, kb_relations, neg_samples_kb, relations, opts):
    neg_samples   = opts.neg_samples
    vect_dim      = opts.vect_dim
    num_entities  = opts.num_entities
    num_relations = opts.num_relations
    l2_reg_entities = opts.l2_entity    
    # +1 for the OOV embedding.
    entities  = Embedding(output_dim=vect_dim, input_dim=num_entities+1, init='normal',name = 'entity_embeddings_DM', W_regularizer=l2(l2_reg_entities))

    entity_vectors = entities(kb_entities)
    entity_negative_vectors = entities(neg_samples_kb)
    relation_vectors = Flatten()(relations(kb_relations))


    get_cross_1 = get_cross(0, neg_samples)
    e1_cross_e2_prime = merge([entity_vectors, entity_negative_vectors], mode = get_cross_1, output_shape = (neg_samples, vect_dim))
    e1_cross_e2 = Lambda(cross_e1_e2, output_shape = (vect_dim,))(entity_vectors)

    score_DM = merge([relation_vectors, e1_cross_e2], mode = lambda X : T.batched_dot(X[0], X[1]), output_shape=())
    score_DM_e2_corrupted = merge([relation_vectors, e1_cross_e2_prime], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

 
    if opts.add_loss:
        get_cross_2 = get_cross(1, neg_samples)
        e1_prime_cross_e2 = merge([entity_vectors, entity_negative_vectors], mode = get_cross_2, output_shape = (neg_samples, vect_dim))
        score_DM_e1_corrupted = merge([relation_vectors, e1_prime_cross_e2], mode = 'dot', output_shape=(neg_samples,), dot_axes=(1,2))

    else:
        score_DM_e1_corrupted = None


    return score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted


def neural_model(MF_data, DM_data, aux_features, opts):
    score_MF, score_MF_corrupted = MF_data
    score_DM, score_DM_e2_corrupted = DM_data
    neg_samples = opts.neg_samples

    stacked_scores = merge([score_MF, score_DM], mode = lambda X: T.stack([X[0], X[1]], axis=1), output_shape=(None,2))

    # tile the (None, 4) feature matrix into a (None, 4, neg_samples) feature matrix by repeating it along the 3rd axis and then shuffle 2nd and 3rd dimension to get a (None, neg_samples, 4)
    aux_features_tiled = Lambda(lambda X : T.tile(X.dimshuffle(0,1,'x'), reps=(1,1,neg_samples)).dimshuffle(0,2,1), output_shape = (neg_samples, 4))(aux_features)

    # stack DM score, MF score, and the 4 features in a single (None, 6) matrix
    stacked_inputs = merge([stacked_scores, aux_features], mode= lambda X: T.concatenate([X[0], X[1]], axis=1), output_shape=(None,6))

    # stack (None, neg_samples, 1) DM and MF negative sample data along with (None, neg_samples, 4) 
    stacked_inputs_e2_corrupt = merge([score_MF_corrupted, score_DM_e2_corrupted, aux_features_tiled], lambda X : T.concatenate([X[0].dimshuffle(0,1,'x'), X[1].dimshuffle(0,1,'x'), X[2]], axis=2), output_shape=(None,neg_samples,6))  

    #define the neural network for outputing new corrupted and real scores
    nnet_layer_1 = Dense(10, init='glorot_normal',activation='tanh', name='dense_1')
    nnet_layer_2 = Dense(10, init='glorot_normal',activation='tanh', name='dense_2')
    nnet_layer_3 = Dense(1, init = 'glorot_normal', name='dense_3')

    layers = [nnet_layer_1, nnet_layer_2, nnet_layer_3]
    # create a function to perform forward passes over the neural network
    forward_pass = get_forward_pass(layers)
    forward_pass_distributed = get_forward_pass([TimeDistributed(layer) for layer in layers])

    score_combined = forward_pass(stacked_inputs)
    score_combined_e2_corrupt = Flatten()(forward_pass_distributed(stacked_inputs_e2_corrupt))

    return score_combined, score_combined_e2_corrupt

def featureNet_model(MF_data, DM_data, aux_features, opts):
    score_MF, score_MF_corrupted = MF_data
    score_DM, score_DM_e2_corrupted = DM_data
    neg_samples = opts.neg_samples

    # tile the (None, 4) feature matrix into a (None, 4, neg_samples) feature matrix by repeating it along the 3rd axis and then shuffle 2nd and 3rd dimension to get a (None, neg_samples, 4)
    aux_features_tiled = Lambda(lambda X : T.tile(X.dimshuffle(0,1,'x'), reps=(1,1,neg_samples)).dimshuffle(0,2,1), output_shape = (neg_samples, 4))(aux_features)

    #define the neural network for outputing new corrupted and real scores
    nnet_layer_1 = Dense(10, init='glorot_normal',activation='tanh',name='dense_1')
    nnet_layer_2 = Dense(10, init='glorot_normal',activation='tanh',name='dense_2')
    nnet_layer_3 = Dense(1, init = 'glorot_normal', activation = 'sigmoid', name='dense_3')

    layers = [nnet_layer_1, nnet_layer_2, nnet_layer_3]
    # create a function to perform forward passes over the neural network
    forward_pass = get_forward_pass(layers)
    forward_pass_distributed = get_forward_pass([TimeDistributed(layer) for layer in layers])

    f1 = forward_pass(aux_features)
    f2 = Flatten()(forward_pass_distributed(aux_features_tiled))

    if opts.normalize_score:
        #nnet_layer_std = Dense(1, init='glorot_normal',name='normalize_MF', W_constraint = maxnorm(0.0001), b_constraint = maxnorm(0.0001))
        nnet_layer_std = Dense(1, init='glorot_normal',name='normalize_MF')
        forward_pass_std = get_forward_pass([nnet_layer_std])
        forward_pass_distributed_std = get_forward_pass([TimeDistributed(nnet_layer_std)])
        score_MF = Lambda(lambda X: (X.dimshuffle(0,'x')), output_shape=(1,))(score_MF)#batch_size x 1
        #(None, neg-samples) --> (None, neg-samples, 1)
        score_MF_corrupted = Lambda(lambda X: X.dimshuffle(0,1,'x'), output_shape=(neg_samples,1))(score_MF_corrupted)        
        score_MF = forward_pass_std(score_MF)
        score_MF = Lambda(lambda X: X[:,0], output_shape=())(score_MF)
        score_MF_corrupted = Flatten()(forward_pass_distributed_std(score_MF_corrupted))
      
        #nnet_layer_std_DM = Dense(1, init='glorot_normal',name='normalize_DM', W_constraint = maxnorm(0.0001), b_constraint = maxnorm(0.0001))
        nnet_layer_std_DM = Dense(1, init='glorot_normal',name='normalize_DM')
        forward_pass_std_DM = get_forward_pass([nnet_layer_std_DM])
        forward_pass_distributed_std_DM = get_forward_pass([TimeDistributed(nnet_layer_std_DM)]) 
        score_DM = Lambda(lambda X: X.dimshuffle(0,'x'), output_shape=(1,))(score_DM)
        score_DM_e2_corrupted = Lambda(lambda X: X.dimshuffle(0,1,'x'), output_shape=(neg_samples,1))(score_DM_e2_corrupted)
        score_DM = forward_pass_std_DM(score_DM)
        score_DM = Lambda(lambda X: X[:,0], output_shape=())(score_DM)
        score_DM_e2_corrupted = Flatten()(forward_pass_distributed_std_DM(score_DM_e2_corrupted))
        
    score_MF = merge([score_MF, f1], lambda X: X[0]*(X[1][:,0]), output_shape = (None,), name="alpha_MF")
    score_MF_corrupted = merge([score_MF_corrupted, f2], lambda X: X[0]*X[1], output_shape = (None, neg_samples))

    score_combined = merge([score_MF, score_DM], lambda X: X[0] + X[1], output_shape = (None,)) 
    score_combined_e2_corrupt = merge([score_MF_corrupted, score_DM_e2_corrupted], mode='sum')

    return score_combined, score_combined_e2_corrupt

def adder_model(MF_data, DM_data, opts):
    score_MF, score_MF_corrupted = MF_data
    score_DM, score_DM_e2_corrupted = DM_data
    neg_samples = opts.neg_samples
    
    if opts.static_alpha:
        score_MF = Lambda(lambda X: opts.alphaMF * X[0], output_shape=())(score_MF)
        score_MF_corrupted = Lambda(lambda X: opts.alphaMF * X[:,], output_shape=(neg_samples,))(score_MF_corrupted)

    score_combined = merge([score_MF, score_DM], lambda X: X[0] + X[1], output_shape = ()) 
    score_combined_e2_corrupt = merge([score_MF_corrupted, score_DM_e2_corrupted], mode='sum')

    return score_combined, score_combined_e2_corrupt



# == Main Driver function for creating a hybrid TF + MF model. Based on the flags, creates all the models in Table-7 of https://arxiv.org/abs/1706.00637

def build_joint_model(opts, combine_func, adder_model=False, add_loss = False):
    neg_samples = opts.neg_samples
    optimizer = opts.optimizer
    vect_dim  = opts.vect_dim
    num_relations = opts.num_relations
    l2_reg_relation = opts.l2_relation   

    # (e1,e2) for distMult
    kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
    # r id for both distMult and Matrix Factorization
    kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
    # (e1,e2) for MF (represented as a single ID)
    kb_entity_pairs = Input(shape=(1,), dtype='int32', name='kb_entity_pairs')
    # negative samples for DM
    neg_samples_kb_MF  = Input(shape=(neg_samples,), dtype = 'int32', name='kb_neg_examples_MF')
    # negative samples for MF
    if add_loss:
        neg_samples_kb_DM  = Input(shape=(neg_samples*2,), dtype = 'int32', name='kb_neg_examples_DM')
    else:
        neg_samples_kb_DM  = Input(shape=(neg_samples,), dtype = 'int32', name='kb_neg_examples_DM')

    if not adder_model:
        aux_features = Input(shape=(4,), dtype='float32', name='auxiliary features')


    flag_hybrid_naive = (opts.model == "FE" or opts.model == "DMFE")
    # score_kb = sigmoid(r_mf.e12 + r_s*e1 + r_o*e2)
    if flag_hybrid_naive:
        relations = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings')

        score_MF, score_MF_corrupted = getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb_MF, relations, opts)
        score_E, score_E_e1_corrupted, score_E_e2_corrupted = getE_score_joint(kb_entities, kb_relations, neg_samples_kb_DM, opts)

        if add_loss:
            print "add loss of MF and E model!!"
            log_likelihood_kb_MF = merge([score_MF, score_MF_corrupted], mode= get_loss_fn(opts), output_shape=(1,))
            log_likelihood_kb_E_2 = merge([score_E, score_E_e2_corrupted], mode= get_loss_fn(opts), output_shape=(1,))
            log_likelihood_kb_E_1 = merge([score_E, score_E_e1_corrupted], mode= get_loss_fn(opts), output_shape=(1,))
            log_likelihood_kb_E = merge([log_likelihood_kb_E_1 , log_likelihood_kb_E_2], lambda X: X[0] + X[1], output_shape = (1,))
            
        if opts.model == "DMFE":
            print "add DM loss"
            relations_DM = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_DM')
            score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted = getDM_score_joint(kb_entities, kb_relations, neg_samples_kb_DM, relations_DM, opts)
            if add_loss:
                log_likelihood_kb_DM_1 = merge([score_DM, score_DM_e1_corrupted], mode= get_loss_fn(opts), output_shape=(1,))
                log_likelihood_kb_DM_2 = merge([score_DM, score_DM_e2_corrupted], mode= get_loss_fn(opts), output_shape=(1,))
                log_likelihood_kb_DM = merge([log_likelihood_kb_DM_1 , log_likelihood_kb_DM_2], lambda X: X[0] + X[1], output_shape = (1,))
                log_likelihood_kb_E =  merge([log_likelihood_kb_DM, log_likelihood_kb_E], mode = lambda X: X[0]+X[1], output_shape=(1,))

            else:
                score_E =  merge([score_DM, score_E], mode = lambda X: X[0]+X[1], output_shape=())
                score_E_e2_corrupted =  merge([score_DM_e2_corrupted, score_E_e2_corrupted], mode = lambda X: X[0]+X[1], output_shape=(neg_samples,))


        if add_loss:
            print "Add loss model"
            log_likelihood_kb =  merge([log_likelihood_kb_MF, log_likelihood_kb_E], mode = lambda X: X[0]+X[1], output_shape=(1,))

        else:
            score_combined = merge([score_MF, score_E], mode = lambda X: X[0]+X[1], output_shape=(1,))
            score_combined_e2_corrupt = merge([score_MF_corrupted, score_E_e2_corrupted], mode = lambda X: X[0]+X[1], output_shape = (neg_samples,)) 

            log_likelihood_kb = merge([score_combined, score_combined_e2_corrupt], mode= softmax_approx, output_shape=(1,))            

               
    else:
        relations_DM = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_DM')
        print("regularizing MF relation embeddings with l2 penalty of: %5.4f" %opts.l2_relation_MF)
        relations_MF = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_MF', W_regularizer=l2(opts.l2_relation_MF))
        score_MF, score_MF_corrupted = getMF_score_joint(kb_entity_pairs, kb_relations, neg_samples_kb_MF, relations_MF, opts)
        score_DM, score_DM_e1_corrupted, score_DM_e2_corrupted = getDM_score_joint(kb_entities, kb_relations, neg_samples_kb_DM, relations_DM, opts)
        if add_loss:
            log_likelihood_kb_MF = merge([score_MF, score_MF_corrupted], mode= softmax_approx, output_shape=(1,))
            log_likelihood_kb_DM_1 = merge([score_DM, score_DM_e1_corrupted], mode= softmax_approx, output_shape=(1,))
            log_likelihood_kb_DM_2 = merge([score_DM, score_DM_e2_corrupted], mode= softmax_approx, output_shape=(1,))
            log_likelihood_kb_DM = merge([log_likelihood_kb_DM_1 , log_likelihood_kb_DM_2], lambda X: X[0] + X[1], output_shape = (1,))
            log_likelihood_kb    = merge([log_likelihood_kb_MF , log_likelihood_kb_DM], lambda X: X[0] + X[1], output_shape = (1,))
        
        else:
            print "Hybrid model from Singh et al. !!"
            score_combined = merge([score_MF, score_DM], mode = lambda X: X[0]+X[1], output_shape=(1,))
            score_combined_e2_corrupt = merge([score_MF_corrupted, score_DM_e2_corrupted], mode = lambda X: X[0]+X[1], output_shape = (neg_samples,)) 
            log_likelihood_kb = merge([score_combined, score_combined_e2_corrupt], mode=get_loss_fn(opts), output_shape=(1,))


    alg = get_optimizer(opts)

    if not adder_model:
        model = Model(input=[kb_entities, kb_entity_pairs, kb_relations, aux_features, neg_samples_kb_DM, neg_samples_kb_MF], output=log_likelihood_kb)
    else:
        model = Model(input=[kb_entities, kb_entity_pairs, kb_relations, neg_samples_kb_DM, neg_samples_kb_MF], output=log_likelihood_kb)
        
    model.compile(loss = lossFn, optimizer=alg)
        
    return model

