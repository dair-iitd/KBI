import warnings,pickle
warnings.simplefilter("ignore", UserWarning)
from keras.layers import *
from lib.evalComplex_OOV import *
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

print "@00"

def get_model_name(opts):
    model_name = opts.model + "_xreg%5.4f_unitNorm%d" %(opts.l2_entity_pair,opts.unit_norm_reg)
    if opts.static_alpha:
        model_name += "_alpha%5.4f_" %opts.alphaMF
    if not opts.shared_r:
        model_name += "differentR"
    if opts.normalize_score:
        model_name += "_normalizedScore_"
    if opts.add_loss:
        model_name += "_addLoss_"
    if not opts.vect_dim == 500:
        model_name += "_dim%5d_" %(opts.vect_dim)

    return model_name


def generator_random(features, labels, opts):
    train_entities, train_relations, neg_samples_data = features;#
    fixed =0 #for aligned AS scores
    if not fixed: del neg_samples_data##CHECK!!
    train_entities_MF = train_entities[0]; train_entities_DM=train_entities[1]
    batch_size = opts.batch_size
    net_data_points = train_entities[0].shape[0]
    batch_per_epoch = int(net_data_points/batch_size)
    neg_samples = opts.neg_samples
    if opts.oov_train:
        neg_samples -= 1
    while True:
        for batch_num in xrange(batch_per_epoch):
            batch_train_entities_MF = [];batch_train_entities_DM = [];batch_train_relations = [];batch_neg_samples_MF = [];batch_neg_samples_DM = []; batch_labels = []
            batch_idx = np.random.randint(0,net_data_points,batch_size)
            for i in xrange(batch_size):
                data_index = batch_idx[i]
                batch_train_entities_DM.append(train_entities_DM[data_index]);batch_train_entities_MF.append(train_entities_MF[data_index])
                batch_train_relations.append(train_relations[data_index])
                batch_labels.append(labels[i])
                 
                if fixed: 
                    neg_data_MF = neg_samples_data[0][data_index]#
                    neg_data_DM = neg_samples_data[1][data_index]
                else:
                    neg_data_MF = np.random.randint(0,opts.num_entity_pairs,neg_samples)
                    if opts.add_loss:
                        neg_data_DM = np.random.randint(0,opts.num_entities,neg_samples*2)
                    else:
                        neg_data_DM = np.random.randint(0,opts.num_entities,neg_samples)
                if opts.oov_train:
                    if not fixed:
                        neg_data_MF = np.concatenate([neg_data_MF,opts.num_entity_pairs*np.ones(1)])
                    
                        neg_column = (opts.num_entities)*np.ones(1)
                        if opts.add_loss:
                            neg_samples_set1 = np.concatenate([neg_data_DM[0: neg_samples], neg_column])
                            neg_samples_set2 = np.concatenate([neg_data_DM[neg_samples:], neg_column])
                            neg_data_DM = np.concatenate([neg_samples_set1, neg_samples_set2])
                        else:
                            neg_data_DM = np.concatenate([neg_data_DM, neg_column])

                batch_neg_samples_MF.append(neg_data_MF)
                batch_neg_samples_DM.append(neg_data_DM)
            batch_train_entities_DM = np.array(batch_train_entities_DM)
            batch_train_entities_MF = np.array(batch_train_entities_MF)
            batch_train_relations = np.array(batch_train_relations)
            batch_labels = np.array(batch_labels)
            batch_neg_samples_DM = np.array(batch_neg_samples_DM)
            batch_neg_samples_MF = np.array(batch_neg_samples_MF)

            batch_features = [batch_train_entities_DM,batch_train_entities_MF,batch_train_relations,batch_train_relations,batch_neg_samples_DM,batch_neg_samples_MF]
            yield batch_features, batch_labels

def generator_random_DME(features, labels, opts):
    train_entities, train_relations, neg_samples_data = features;#
    fixed =0 #for aligned AS scores
    if not fixed: del neg_samples_data##CHECK!!
    train_entities_DM=train_entities
    batch_size = opts.batch_size
    net_data_points = train_entities.shape[0]
    batch_per_epoch = int(net_data_points/batch_size)
    neg_samples = opts.neg_samples
    if opts.oov_train:
        neg_samples -= 1
    while True:
        for batch_num in xrange(batch_per_epoch):
            batch_train_entities_DM = [];batch_train_relations = [];batch_neg_samples_DM = []; batch_labels = []
            batch_idx = np.random.randint(0,net_data_points,batch_size)
            for i in xrange(batch_size):
                data_index = batch_idx[i]
                batch_train_entities_DM.append(train_entities_DM[data_index])
                batch_train_relations.append(train_relations[data_index])
                batch_labels.append(labels[i])

                if fixed:
                    neg_data_DM = neg_samples_data[1][data_index]
                else:
                    if opts.add_loss:
                        neg_data_DM = np.random.randint(0,opts.num_entities,neg_samples*2)
                    else:
                        neg_data_DM = np.random.randint(0,opts.num_entities,neg_samples)
                if opts.oov_train:
                    if not fixed:
                        neg_column = (opts.num_entities)*np.ones(1)
                        if opts.add_loss:
                            neg_samples_set1 = np.concatenate([neg_data_DM[0: neg_samples], neg_column])
                            neg_samples_set2 = np.concatenate([neg_data_DM[neg_samples:], neg_column])
                            neg_data_DM = np.concatenate([neg_samples_set1, neg_samples_set2])
                        else:
                            neg_data_DM = np.concatenate([neg_data_DM, neg_column])

                batch_neg_samples_DM.append(neg_data_DM)
            batch_train_entities_DM = np.array(batch_train_entities_DM)
            batch_train_relations = np.array(batch_train_relations)
            batch_labels = np.array(batch_labels)
            batch_neg_samples_DM = np.array(batch_neg_samples_DM)

            batch_features = [batch_train_entities_DM,batch_train_relations,batch_neg_samples_DM]
            yield batch_features, batch_labels


def train_joint_DME(model, modelEvalCallback, opts, train_entities_DM, train_relations, neg_samples_DM):
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities_DM.shape[0]

    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    dummy_y = np.zeros(trainDataSize)

    print("DM neg samples:", neg_samples_DM.shape)

    steps_per_epoch_val = int(trainDataSize/opts.batch_size)
    print train_entities_DM, train_relations,neg_samples_DM
    history = model.fit_generator(generator_random_DME([train_entities_DM, train_relations,neg_samples_DM], dummy_y, opts), epochs=opts.nb_epochs, callbacks=callback_functions, steps_per_epoch = steps_per_epoch_val)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr)
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits)
    print("="*50)


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
        steps_per_epoch_val = int(trainDataSize/opts.batch_size)#1
        history = model.fit_generator(generator_random([train_entities, train_relations,neg_samples], dummy_y, opts), epochs=opts.nb_epochs, callbacks=callback_functions, steps_per_epoch = steps_per_epoch_val)
    else:
        history = model.fit(x=[train_entities[1], train_entities[0], train_relations, frequency_features, neg_samples[1], neg_samples[0]], y=dummy_y, batch_size=opts.batch_size, nb_epoch=opts.nb_epochs, callbacks = callback_functions)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)   


def get_embeddings(model):
    embeddings = {}
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'relation_embeddings':
            embeddings[layer.name] = layer.get_weights()[0]

    return embeddings

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

def load_pretrained_complex(model,opts):
    model_file    = "final-complex-log/models/complex_fb15k_logistic_100dim.h5"
    print "Initializing model with file: %s" %model_file
    f             = h5py.File(model_file,"r")
    entity_im_init   = f['entity_embeddings_im']['entity_embeddings_im']['embeddings'].value
    entity_real_init   = f['entity_embeddings_real']['entity_embeddings_real']['embeddings'].value
    relation_im_init = f['relation_embeddings_im']['relation_embeddings_im']['embeddings'].value
    relation_real_init = f['relation_embeddings_real']['relation_embeddings_real']['embeddings'].value
    model_file    = "final-complex-log/models/MF_fb15k_logistic.h5"
    print "Initializing model with file: %s" %model_file
    f             = h5py.File(model_file,"r")
    entity_init   = f['entity_embeddings']['entity_embeddings']['embeddings'].value
    relation_init = f['relation_embeddings']['relation_embeddings']['embeddings'].value
    for layer in model.layers:
        if layer.name == 'entity_embeddings_MF':
            layer.set_weights([entity_init])
        elif layer.name == 'relation_embeddings_MF':
            layer.set_weights([relation_init])
        elif layer.name == 'entity_embeddings_im':
            layer.set_weights([entity_im_init])
        elif layer.name == 'relation_embeddings_im':
            layer.set_weights([relation_im_init])
        elif layer.name == 'entity_embeddings_real':
            layer.set_weights([entity_real_init])
        elif layer.name == 'relation_embeddings_real':
            layer.set_weights([relation_real_init])

    return model

######################################################### MAIN DRIVER FUNCTION ##########################################################

 
if __name__ == '__main__':
    opts = get_params() ;
    print opts
    if opts.train:
        ''' get train data stuff '''
        if opts.model == "DME":
            dummy_data = np.array([0])
            train_entities_MF, train_relations, train_pair_ids, neg_samples_MF = (dummy_data,dummy_data,dummy_data,dummy_data)
        else:
            train_entities_MF, train_relations, train_pair_ids, neg_samples_MF = get_train_data_matrix(opts, verbose=True)#?
        train_entities_DM, train_relations, neg_samples_DM = get_train_data_tensor(opts, verbose=True)
        ''' If we are not adding losses then use aligned negative samples '''
        if not opts.add_loss:
            neg_samples_DM, neg_samples_MF = get_negative_samples_joint(opts, verbose=True)
        print("Done getting neg samples")

    if opts.oov_train:
        if opts.train:
            trainDataSize = neg_samples_DM.shape[0]
            ''' convert (None, 400) to (None, 402) by appending an OOV after the first 200, then at the end'''
            if opts.add_loss:
                neg_column = (opts.num_entities)*np.ones(trainDataSize)
                neg_samples_set1 = np.c_[neg_samples_DM[:, 0: opts.neg_samples], neg_column]
                neg_samples_set2 = np.c_[neg_samples_DM[:, opts.neg_samples:], neg_column]
                neg_samples_DM = np.c_[neg_samples_set1, neg_samples_set2] 
            else:
                neg_samples_DM = np.c_[neg_samples_DM, (opts.num_entities)*np.ones(trainDataSize)]
            if not opts.model == "DME":
                neg_samples_MF = np.c_[neg_samples_MF, (opts.num_entity_pairs)*np.ones(trainDataSize)]
        opts.neg_samples += 1


    if not opts.model == "adderNet":
        frequency_features, frequency_features_test = get_other_features(opts)
        mu = np.mean(frequency_features, axis=0)
        sigma = np.std(frequency_features, axis=0)
        frequency_features = (frequency_features - mu) / sigma
        frequency_features_test = (frequency_features_test - mu) / sigma 

        print("Done getting features")
    else:
        frequency_features = frequency_features_test = 0
    ''' get test data stuff '''
    testData_DM, oov_flags_1, oov_flags_2, filter_e2_DM, oov_e2 = get_test_data_DM(opts)
    print("got data for DM")
    if opts.model == "DME":
        testData_MF, oov_flags, seen_e2_MF, oov_counts = (0,0,0,0)
    else:
        testData_MF, oov_flags, seen_e2_MF, oov_counts = get_test_data_matrix(opts, verbose=True)
    

    ''' partitions the candidate set into 3 disjoint sets: e2s such that (e1,e2) is trained, and e2s such that (e1,e2) is OOV but e2 is trained, and OOV e2s
        call the 3 sets of e2s set1_e2, set2_e2 and set3_e2. Note that we do not include filtered e2s in any of the 3 sets. (also note that set3_e2 is precisely oov_e2)
    '''
    print "Ready to convert"
    file_name_1 = "precompute/precompute_set1_e2.pickle";file_name_2 = "precompute/precompute_set2_e2.pickle"
    def convert_int(data):
        x=[]
        for ele in data:
            if ele[0]:
                x.append([int(ele_i) for ele_i in ele])
            else:
                x.append([])
        return x
    set1_e2 = []
    if not opts.model == "DME":
        set1_e2, set2_e2 = convertToSets(opts, seen_e2_MF, filter_e2_DM)
    
    print "Conversion done"
    num_e2_compared = np.zeros(len(set1_e2))
    print "Ready to build"
   
    if not opts.model == "adderNet" and not opts.model == "DME": 
        s=0
        number_e2_seen = []
        oovs = 0.0
        for i in xrange(len(set1_e2)):
            curr_e2_seen = len(set1_e2[i])
            s += curr_e2_seen
            number_e2_seen.append(curr_e2_seen)
            oovs += oov_flags[i]    
        print(oovs)
        oovs /= oov_flags.shape[0]
        print("OOV rate: ",100.0*oovs)
        print(Counter(number_e2_seen))
        print("average E_2: ", s/len(set1_e2))

    if opts.model == "DME":
        model = build_joint_model(opts, None)
        evalFunc = lambda model: score_DME(model, opts, testData_DM, oov_flags_1, oov_flags_2, oov_e2, filter_e2_DM, model_eval_DME())

    elif opts.model == "adderNet":
        # score_fact_i = MF + DM
        if opts.add_loss:
            print "Add Loss Model!"
            model = build_joint_model(opts, adder_model, True, add_loss = True)
        else:
            print "SS Model!"
            model = build_joint_model(opts, adder_model, True)
        if opts.use_complex:
            evalFunc = lambda model: score_adder(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, adder_eval_complex())
        else:
            evalFunc = lambda model: score_adder(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, adder_eval())
        if opts.init_model:
            model = load_pretrained_complex(model,opts)
    elif opts.model == "FE" or opts.model == "DMFE":
        if opts.add_loss:
            model = build_joint_model(opts, None, add_loss = True)
        else:
            model = build_joint_model(opts, None)
            
        if opts.model == "FE":
            evalFunc = lambda model: score_FE(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_2, oov_flags, oov_e2, model_eval_FE())
        else:
            evalFunc = lambda model: score_DMFE(model, opts, testData_DM, testData_MF, seen_e2_MF, set1_e2, set2_e2, oov_flags_1, oov_flags_2, oov_flags, oov_e2, model_eval_DMFE())

    print(model.summary())
    if (not opts.train):

        if opts.model_path != "":
            model_file = "finalModels/%s" %opts.model_path
        else:
            print("no trained model provided!");exit()
        model_file = "eval-model/%s" %opts.model_path
        print "Loading weights from file %s" %model_file
        model.load_weights(model_file, by_name=True)
        mrr, hits = evalFunc(model)
        print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))
        mrr_DM, hits_DM = scoreDM_OOV(model, opts, testData_DM, None, oov_flags_1, oov_flags_2, filter_e2_DM, oov_e2, distMult_oovEval(), None)

        print("MRR_DM: %5.4f, HITS_DM: %5.4f" %(mrr_DM, hits_DM))

    else:
        model_name = get_model_name(opts)
        modelEvalCallback = EvaluateModel(evalFunc, opts, model_name, True)
        if opts.model == "DME":
            train_joint_DME(model, modelEvalCallback, opts, train_entities_DM, train_relations, neg_samples_DM)
        else:
            train_joint(model, modelEvalCallback, opts, [train_entities_MF, train_entities_DM], train_relations, [neg_samples_MF, neg_samples_DM], frequency_features)
