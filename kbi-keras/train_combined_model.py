import warnings
warnings.simplefilter("ignore", UserWarning)
from keras.layers import *
from lib.wrappers import *
from lib.combinedModel import *
from lib.evalJoint import *
from lib.evalJoint_adder import *
from lib.eval_modelFE import *
from lib.evalTensor_OOV import *
from lib.eval_modelDMFE import *
from keras.callbacks import *
from helpers import *
from get_params import *
from collections import Counter
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.generic_utils import Progbar
import theano
import theano.tensor as T
import numpy as np
import joblib
import cPickle
import sys
import resource
import h5py


# == A helper to create a model name from cmd line options

def get_model_name(opts):
    model_name = opts.model + "_xreg%5.4f_unitNorm%d" %(opts.l2_entity_pair,opts.unit_norm_reg)
    if opts.static_alpha:
        model_name += "_alpha%5.4f_" %opts.alphaMF
    if opts.static_beta:
        model_name += "_beta_"

    if not opts.shared_r:
        model_name += "differentR"
    if opts.normalize_score:
        model_name += "_normalizedScore_"

    if opts.add_loss:
        model_name += "_addLoss_"

    if opts.dropout_DM + opts.dropout_MF:
        model_name += "_dropout%2.4fDM_%2.4fMF_" %(opts.dropout_DM, opts.dropout_MF)

    if not opts.vect_dim == 500:
        model_name += "_dim%5d_" %(opts.vect_dim)

    return model_name


# == Return numpy arrays representing entity and relation embeddings from a keras model (either MF or TF)

def get_embeddings(model):
    embeddings = {}
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'relation_embeddings':
            embeddings[layer.name] = layer.get_weights()[0]

    return embeddings



# == given a set of options, load in pretrained MF and TF weights. helpful for training a joint model with pretraining

def load_pretrained(model,opts):
    model_MF = build_MFModel(opts)
    _, model_DM = build_linear_distMult(opts)
    model_DM.load_weights("bestModels/best_distMult_NYT_FB.h5")
    model_MF.load_weights("bestModels/best_MF_NYT_FB.h5")

    MF_embeddings = get_embeddings(model_MF)
    DM_embeddings = get_embeddings(model_DM)

    for layer in model.layers:
        if layer.name == 'entity_embeddings_DM':
            layer.set_weights([DM_embeddings['entity_embeddings']])
        elif layer.name == 'entity_embeddings_MF':
            layer.set_weights([MF_embeddings['entity_embeddings']])
        elif layer.name == 'relation_embeddings_DM':
            layer.set_weights([DM_embeddings['relation_embeddings']])

    return model





# == Entry function for training a joint MF-TF model. 

def train_joint(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples, frequency_features):   
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities[0].shape[0]

    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    if opts.dropout_DM + opts.dropout_MF:
        print("Dropout: entity vectors --> 0 after every update %5.4f times. Entity pair vectors --> 0 after every update %5.4f times.") %(opts.dropout_DM, opts.dropout_MF)
        callback_functions.append(Dropout_e(opts))

    dummy_y = np.zeros(trainDataSize)

    print("after adding OOV negative sample")
    print("DM neg samples:", neg_samples[1].shape)
    print("MF neg samples:", neg_samples[0].shape)

    if opts.model == 'adderNet':
        history = model.fit(x=[train_entities[1], train_entities[0], train_relations, neg_samples[1], neg_samples[0]], y=dummy_y, batch_size=opts.batch_size, nb_epoch=opts.nb_epochs, callbacks = callback_functions)
    else:
        history = model.fit(x=[train_entities[1], train_entities[0], train_relations, frequency_features, neg_samples[1], neg_samples[0]], y=dummy_y, batch_size=opts.batch_size, nb_epoch=opts.nb_epochs, callbacks = callback_functions)


    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)   





def dispatch_model(opts, testData, oov_sets, oov_flag_list, oov_e2, frequency_features_test):
    testData_DM, testData_MF = testData
    set1_e2, set2_e2 = oov_sets
    oov_flags_1, oov_flags_2, oov_flags = oov_flag_list

    if opts.model == "joint":
        # for fact_i we get <score_DM, score_MF, f1, f2, f3, f4> --> 3 layer FCN --> score
        model = build_joint_model(opts, neural_model)
        evalFunc = lambda model: score_joint(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, frequency_features_test, joint_eval())

    elif opts.model == "featureNet":
        # score_fact_i = \alpha MF + DM || \alpha is obtained from 3 layer FCN, which takes in all 4 features 
        model = build_joint_model(opts, featureNet_model)
        evalFunc = lambda model: score_joint(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, frequency_features_test, featureNet_eval())

    elif opts.model == "featureNet2":
        # score_fact_i = \alpha MF + (1.0 - \alpha) DM || \alpha is obtained from 3 layer FCN, which takes in all 4 features 
        model = build_joint_model(opts, featureNet_model)
        evalFunc = lambda model: score_joint(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, frequency_features_test, featureNet2_eval())

    elif opts.model == "adderNet":
        # score_fact_i = MF + DM
        if opts.add_loss:
            model = build_joint_model(opts, adder_model, True, add_loss = True)
        else:
            model = build_joint_model(opts, adder_model, True)

        evalFunc = lambda model: score_adder(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, adder_eval())

    elif opts.model == "FE" or opts.model == "DMFE":
        if opts.add_loss:
            model = build_joint_model(opts, None, add_loss = True)
        else:
            model = build_joint_model(opts, None)
            
        if opts.model == "FE":
            evalFunc = lambda model: score_FE(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_2, oov_flags, oov_e2, model_eval_FE())
        else:
            evalFunc = lambda model: score_DMFE(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, model_eval_DMFE())


    return model, evalFunc







######################################################### MAIN DRIVER FUNCTION ##########################################################

 
if __name__ == '__main__':
    opts = get_params() 
    if opts.train:
        ''' get train data stuff '''
        train_entities_MF, train_relations, train_pair_ids, neg_samples_MF = get_train_data_matrix(opts, verbose=True)#?
        train_entities_DM, _, neg_samples_DM = get_train_data_tensor(opts, verbose=True)
        ''' If we are not adding losses then use aligned negative samples '''
        if not opts.add_loss:
            neg_samples_DM, neg_samples_MF = get_negative_samples_joint(opts, verbose=True)
        print("Precomputed negative samples read.")

    if opts.oov_train:
        if opts.train:
            trainDataSize = neg_samples_MF.shape[0]
            ''' convert (None, 400) to (None, 402) by appending an OOV after the first 200, then at the end'''
            if opts.add_loss:
                neg_column = (opts.num_entities)*np.ones(trainDataSize)
                neg_samples_set1 = np.c_[neg_samples_DM[:, 0: opts.neg_samples], neg_column]
                neg_samples_set2 = np.c_[neg_samples_DM[:, opts.neg_samples:], neg_column]
                neg_samples_DM = np.c_[neg_samples_set1, neg_samples_set2] 
            else:
                neg_samples_DM = np.c_[neg_samples_DM, (opts.num_entities)*np.ones(trainDataSize)]
            neg_samples_MF = np.c_[neg_samples_MF, (opts.num_entity_pairs)*np.ones(trainDataSize)]


        opts.neg_samples += 1


    frequency_features, frequency_features_test = get_other_features(opts)
    mu = np.mean(frequency_features, axis=0)
    sigma = np.std(frequency_features, axis=0)
    frequency_features = (frequency_features - mu) / sigma
    frequency_features_test = (frequency_features_test - mu) / sigma 

    print("frequency features computed.")

    ''' get test data stuff '''
    testData_DM, oov_flags_1, oov_flags_2, filter_e2_DM, oov_e2 = get_test_data_DM(opts)
    testData_MF, oov_flags, seen_e2_MF, oov_counts = get_test_data_matrix(opts, verbose=True)
    
    print("Test data read.")

    ''' partitions the candidate set into 3 disjoint sets: e2s such that (e1,e2) is trained, and e2s such that (e1,e2) is OOV but e2 is trained, and OOV e2s
        call the 3 sets of e2s set1_e2, set2_e2 and set3_e2. Note that we do not include filtered e2s in any of the 3 sets. (also note that set3_e2 is precisely oov_e2)
    '''
    set1_e2, set2_e2 = convertToSets(opts, seen_e2_MF, filter_e2_DM)
    num_e2_compared = np.zeros(len(set1_e2))
    
 
    model, evalFunc = dispatch_model(opts, [testData_DM, testData_MF], [set1_e2, set2_e2], [oov_flags_1, oov_flags_2, oov_flags], oov_e2, frequency_features_test)

    print(model.summary())

    if (not opts.train):
        if opts.model_path != "":
            model_file = "finalModels/%s" %opts.model_path
        else:    
            model_name = get_model_name(opts)
            model_file = "%s_%s.h5" %(model_name, opts.dataset)
            model_file = model_file.replace("/", "_")
            model_file = "bestModels/%s" %model_file

        model.load_weights(model_file, by_name=True)
        mrr, hits = evalFunc(model)
        print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))
        
    else:
        print("scores before training")
        mrr, hits = evalFunc(model)
        print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))
        model_name = get_model_name(opts)
        modelEvalCallback = EvaluateModel(evalFunc, opts, model_name, True)
        train_joint(model, modelEvalCallback, opts, [train_entities_MF, train_entities_DM], train_relations, [neg_samples_MF, neg_samples_DM], frequency_features)
