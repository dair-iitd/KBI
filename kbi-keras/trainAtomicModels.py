from time import gmtime, strftime
from keras.layers import *
from lib.wrappers import *
from lib.models import *
from lib.MFmodel import *
from lib.evalTensor_OOV import * 
from lib.evalComplex_OOV import *
from lib.evalMF_OOV import *
from keras.callbacks import *
from helpers import *
from get_params import *
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
from collections import Counter
from collections import defaultdict as dd

def generator_random(features, labels, opts):
    train_entities, train_relations, neg_samples_data = features;del neg_samples_data
    batch_size = opts.batch_size
    batch_train_entities = []
    batch_train_relations = []
    batch_neg_samples = []
    batch_labels = []
    net_data_points = train_entities.shape[0]
    batch_per_epoch = int(net_data_points/opts.batch_size)
    neg_samples = opts.neg_samples
    if re.search("^MF", opts.model):
        num_entities = opts.num_entity_pairs
    else:
        num_entities = opts.num_entities
    if opts.oov_train:
        neg_samples -= 1
    while True:
        for batch_num in xrange(batch_per_epoch):
            batch_train_entities = [];batch_train_relations = [];batch_neg_samples = []; batch_labels = []
            batch_idx = np.random.randint(0,net_data_points,batch_size)
            for i in xrange(batch_size):
                data_index = batch_idx[i]
                batch_train_entities.append(train_entities[data_index])
                batch_train_relations.append(train_relations[data_index])
                batch_labels.append(labels[i])
                if re.search("^MF", opts.model):#opts.model == "MF":
                    neg_data = np.random.randint(0,num_entities,neg_samples)
                else:
                    neg_data = np.random.randint(0,num_entities,neg_samples*2)
                if opts.oov_train:
                    if re.search("^MF", opts.model):
                        neg_data = np.concatenate([neg_data,num_entities*np.ones(1)])
                    else:
                        neg_column = (num_entities)*np.ones(1)
                        neg_samples_set1 = np.concatenate([neg_data[0: neg_samples], neg_column])
                        neg_samples_set2 = np.concatenate([neg_data[neg_samples:], neg_column])
                        neg_data = np.concatenate([neg_samples_set1, neg_samples_set2])

                batch_neg_samples.append(neg_data)

            batch_train_entities = np.array(batch_train_entities)
            batch_train_relations = np.array(batch_train_relations)
            batch_labels = np.array(batch_labels)
            batch_neg_samples = np.array(batch_neg_samples)

            batch_features = [batch_train_entities,batch_train_relations,batch_neg_samples]
            yield batch_features, batch_labels


def train_net(model, aux_model, modelEvalCallback, opts, train_entities, train_relations, neg_samples):
    print "aux_model::", aux_model
    callback_functions = [modelEvalCallback]

    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    if opts.typing:
        print("Multi instance learning with TypeNet.")
        entity_types, entity_neg_types = get_typing_data(opts)
        typeNetTrainer = typeNetTrainerCallback(aux_model, entity_types, entity_neg_types, opts)
        callback_functions.append(typeNetTrainer)

    dummy_y = np.zeros((train_entities.shape[0],1))
    steps_per_epoch_val = int(train_entities.shape[0]/opts.batch_size)#1

    history = model.fit_generator(generator_random([train_entities, train_relations,neg_samples], dummy_y, opts), epochs=opts.nb_epochs, callbacks=callback_functions, steps_per_epoch = steps_per_epoch_val)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)

def int_dict():
    return dd(int)


def train_MF(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples):
    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    dummy_y = np.zeros(train_entities.shape[0])
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities.shape[0]

    steps_per_epoch_val = int(train_entities.shape[0]/opts.batch_size)#1
    history = model.fit_generator(generator_random([train_entities, train_relations, neg_samples], dummy_y, opts), epochs=opts.nb_epochs, callbacks=callback_functions, steps_per_epoch = steps_per_epoch_val)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)   


def train_MF_constraint(model, aux_model, modelEvalCallback, opts, train_entities, train_entity_pairs, train_relations, neg_samples):
    modelEvalCallback = EvaluateModel(evalFunc, opts, "%s_%d" %(opts.model, opts.oov_train), aux_model)
    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())
    
    print("training model to generate ep embedding from constituent e embeddings")
    print "train_entities[:,1].shape:: ", train_entities[:,1].shape
    constraintTrainer = constraintTrainerCallback(aux_model, train_entity_pairs, train_entities[:,0], train_entities[:,1], opts)
    callback_functions.append(constraintTrainer)

    dummy_y = np.zeros(train_entities.shape[0])
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities.shape[0]
    print "Size details: train_entity_pairs, train_relations, neg_samples",train_entity_pairs.shape, train_relations.shape,neg_samples.shape
    steps_per_epoch_val = int(train_entities.shape[0]/opts.batch_size)
    history = model.fit_generator(generator_random([train_entity_pairs, train_relations, neg_samples], dummy_y, opts), epochs=opts.nb_epochs, callbacks=callback_functions, steps_per_epoch = steps_per_epoch_val)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)   
    
def train_MF_constraint2(model, aux_model, modelEvalCallback, opts, train_entities, train_entity_pairs, train_relations, train_relations_DM, neg_samples, neg_samples_DM):
    aux_model_EP, aux_model_DM = aux_model
    modelEvalCallback = EvaluateModel(evalFunc, opts, "%s_%d" %(opts.model, opts.oov_train), aux_model)
    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())
    
    print("training model to generate ep embedding from constituent e embeddings")
    constraintTrainer = constraintTrainerCallback(aux_model_EP, train_entity_pairs, train_entities[:,0], train_entities[:,1], opts)
    callback_functions.append(constraintTrainer)

    print("training DM model with r and shared e embeddings")###
    constraint2Trainer = constraint2TrainerCallback(aux_model_DM, train_entities[:,0], train_entities[:,1], train_relations_DM, neg_samples_DM, opts)
    callback_functions.append(constraint2Trainer)
    
    dummy_y = np.zeros(train_entities.shape[0])
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities.shape[0]
    print "Size details: train_entity_pairs, train_relations, neg_samples",train_entity_pairs.shape, train_relations.shape,neg_samples.shape
    history = model.fit(x=[train_entity_pairs, train_relations, neg_samples], y=dummy_y, batch_size=opts.batch_size, epochs=opts.nb_epochs, callbacks=callback_functions)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)   


def dispatch_model(opts, inputs):
    train_entities, train_relations, neg_samples = inputs

    if not opts.oov_eval and opts.model not in ['MF', 'freq']:
        print("No support for non OOV training")
        sys.exit(1)

    if opts.model == "distMult":
        model = build_atomic_model(opts, getDM_score)
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
        _ , oov_flags, _, _  = get_test_data_matrix(opts, verbose=True)

        score_fn = distMult_oovEval()
        score_fn_aux = distMult_get_scores()
        distMult_scorer = distMult_run()
        evalFunc = lambda model: scoreDM_OOV(model,opts, testData, [train_entities, train_relations], oov_flags_1, oov_flags_2,seen_e2, oov_e2, score_fn, score_fn_aux, distMult_scorer,oov_flags)                
        
        if opts.train and opts.init_model:
            aux_model_file = model_file  = ""
            if re.search("fb15k-237", opts.dataset):
                aux_model_file = "finalModels/distMult_FB15k-237_dim100.h5"#add path here
            elif re.search("fb15k", opts.dataset):
                aux_model_file = "finalModels/distMult_FB15k_dim100_2.h5"
            elif re.search("wn18", opts.dataset):
                aux_model_file = "finalModels/distMult_wn18_dim100_maxmargin.h5"
            elif re.search("nyt-fb", opts.dataset):
                aux_model_file = "finalModels/distMult_NYT_FB_dim100.h5"

            print "Initializing with file: %s and %s" %(model_file, aux_model_file)
            model.load_weights("%s" %(aux_model_file), by_name=True)

    elif opts.model == "typedDM":
        model = build_typed_model(opts, getDM_score)
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
        _ , oov_flags, _, _  = get_test_data_matrix(opts, verbose=True)

        score_fn = typedDM_oovEval()
        evalFunc = lambda model: scoreTDM_OOV(model, opts, testData, [train_entities, train_relations],oov_flags_1, oov_flags_2,seen_e2, oov_e2, score_fn, oov_flags)


    elif opts.model == "complex":
        model = build_atomic_model(opts, getComplex_score)
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
        _ , oov_flags_MF, _, _  = get_test_data_matrix(opts, verbose=True)
        score_fn = complex_oovEval()
        evalFunc = lambda model: score_complex_helper(model, opts, testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, score_fn,oov_flags_MF)     

    elif opts.model == "E":
        model = build_atomic_model(opts, getE_score)
        score_fn  = E_oovEval() 
        testData, oov_flags_1, oov_flags_2,  seen_e2, oov_e2 = get_test_data_DM(opts, True)
        _ , oov_flags, _, _  = get_test_data_matrix(opts, verbose=True)
        evalFunc = lambda model: scoreE_OOV(model,train_entities,  testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, score_fn, oov_flags)
         
    elif opts.model == "TransE":
        model = build_atomic_model(opts, getTransE_score)
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
        score_fn = TransE_oovEval()
        _ , oov_flags, _, _  = get_test_data_matrix(opts, verbose=True)
        evalFunc = lambda model: scoreTransE_OOV(opts, model,testData, train_entities, oov_flags_1, oov_flags_2,seen_e2, oov_e2, score_fn, oov_flags)        
        
    elif opts.model == "MF":
        aux_model = None
        model = build_MFModel(opts)
        testData, oov_flags, seen_e2, oov_counts = get_test_data_matrix(opts, verbose=True)

        # if we're not doing oov evaluation, this is essentially what we are saying
        if not opts.oov_eval:
            oov_flags = np.zeros_like(oov_flags)
            oov_counts = np.zeros_like(oov_counts)

        counter = Counter(train_entities)
        evalFunc = lambda model: scoreMF_OOV_naive(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval_naive()) 
    
    elif opts.model == "MF-constraint":#in paper
        model, aux_model = build_MFModel_eEPContraint(opts)
        testData, oov_flags, seen_e2, oov_counts = get_test_data_matrix(opts, verbose=True)
        testData_DM, _, oov_flags_e2, seen_e2_DM, _ = get_test_data_DM(opts, True)
        mapping_ep_e2 = get_mapping_data_matrix(opts)
        print model.summary()
        print aux_model.summary()
        counter = Counter(train_entities)

        if opts.train and opts.init_model:
            aux_model_file = model_file  = ""
            if re.search("fb15k-237",opts.dataset):
                aux_model_file = "eval-model/distMult_fb15k-237_1.h5"#pre-trained DM model path
                model_file     = "eval-model/MF_fb15k-237_0.h5"#pre-trained MF model path
            elif re.search("fb15k",opts.dataset):
                aux_model_file = "eval-model/distMult_fb15k_0.h5"
                model_file     = "eval-model/MF_fb15k_0.h5"
            elif re.search("wn18",opts.dataset):
                aux_model_file = "eval-model/distMult_wn18_0.h5"
                model_file     = "eval-model/MF_wn18_0.h5"
            elif re.search("nyt-fb",opts.dataset):
                aux_model_file = "eval-model/distMult_nyt-fb_1.h5"
                model_file     = "eval-model/MF_nyt-fb_1.h5"

            print "Initializing with file: %s and %s" %(model_file, aux_model_file)
            aux_model.load_weights("%s" %(aux_model_file), by_name=True)#init e w/t DM        
        
        if opts.train:
            evalFunc = lambda model: scoreMF_OOV_naive(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval_naive()) 
        else:
            evalFunc = lambda model: scoreMF_OOV_generate_OOVep_embedding_fast(model, aux_model, opts, counter, testData, testData_DM, oov_flags, seen_e2, oov_flags_e2, seen_e2_DM, mapping_ep_e2, oov_counts, MF_oovEval_gen())

    elif opts.model == "MF-constraint-2":
        model, aux_model_EP, aux_model_DM = build_MFModel_eEPContraint_DM(opts)
        testData, oov_flags, seen_e2, oov_counts = get_test_data_matrix(opts, verbose=True)
        testData_DM, _, oov_flags_e2, seen_e2_DM, _ = get_test_data_DM(opts, True)
        mapping_ep_e2 = get_mapping_data_matrix(opts)
        print model.summary()
        print aux_model_EP.summary()
        print aux_model_DM.summary()
        counter = Counter(train_entities)

        if opts.train and opts.init_model:
            aux_model_file = model_file  = ""
            print opts.dataset 
            if opts.dataset == "fb15k":
                aux_model_file = "eval-model/distMult_fb15k_0.h5"
                model_file     = "eval-model/MF_fb15k_0.h5"
            elif opts.dataset == "fb15k-237":
                aux_model_file = "eval-model/distMult_fb15k-237_1.h5"
                model_file     = "eval-model/MF_fb15k-237_0.h5"
            elif opts.dataset == "wn18":
                aux_model_file = "eval-model/distMult_wn18_0.h5"
                model_file     = "eval-model/MF_wn18_0.h5"
            elif opts.dataset == "nyt-fb":
                aux_model_file = "eval-model/distMult_nyt-fb_1.h5"
                model_file     = "eval-model/MF_nyt-fb_1.h5"

            print "Initializing with file: %s and %s" %(model_file, aux_model_file)
            model.load_weights("%s" %(model_file), by_name=True)#init MF ep with MF
            #model.load_weights("%s" %(aux_model_file), by_name=True)#init MF r from DiM - Doesn't work!!!!
            aux_model_EP.load_weights("%s" %(aux_model_file), by_name=True)#init e w/t DM  
            aux_model_DM.load_weights("%s" %(aux_model_file), by_name=True)#init e & r w/t DM
        
        if opts.train:
            evalFunc = lambda model: scoreMF_OOV_naive(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval_naive()) 
        else:
            evalFunc = lambda model: scoreMF_OOV_generate_OOVep_embedding_fast(model, aux_model_EP, opts, counter, testData, testData_DM, oov_flags, seen_e2, oov_flags_e2, seen_e2_DM, mapping_ep_e2, oov_counts, MF_oovEval_gen())
        aux_model = [aux_model_EP, aux_model_DM]
    elif opts.model == "MF-replace-with-oldEmbed":#only eval is different
        aux_model = None
        model = build_MFModel(opts)
        testData, oov_flags, seen_e2, oov_counts = get_test_data_matrix(opts, verbose=True)
        testData_DM, _, _, _, _ = get_test_data_DM(opts, True)
        # if we're not doing oov evaluation, this is essentially what we are saying
        if not opts.oov_eval:
            oov_flags = np.zeros_like(oov_flags)
            oov_counts = np.zeros_like(oov_counts)

        counter = Counter(train_entities)
        
        new_embeddings_data = get_new_embeddings_data(opts)
        old_id, old_id_new_embeddings, new_id_new_embeddings, new_e1e2_epId_map = new_embeddings_data
        seen_e2, oov_counts, oov_flags = update_with_new_embedding_data(testData_DM, old_id, new_e1e2_epId_map, seen_e2, oov_counts, oov_flags,opts)
        #To update: entity_weights, seen_e2, oov_counts, oov_flags
        #old_id, old_id_new_embeddings, new_id_new_embeddings

        print("Number of old embeddings updated: %d  \nNumber of new embeddings generated: %d \n" %(len(old_id),len(new_e1e2_epId_map)))

        evalFunc = lambda model: scoreMF_OOV_generate_OOVep_embedding(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval_naive(), old_id, old_id_new_embeddings, new_id_new_embeddings, pretrained=True) 
    elif opts.model == "freq":
        print("Running baseline models")
        pred_array_1 = dd(int_dict) 
        pred_array_2 = dd(int_dict)
        trainDataSize = train_entities.shape[0]
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
        for i in xrange(trainDataSize):
            e1,e2 = train_entities[i] 
            r = train_relations[i]
            pred_array_1[int(r)][int(e2)] += 1
            pred_array_2[int(e1)][int(e2)] += 1

        MRR =[0.0, 0.0]
        hits=[0.0, 0.0]
        hits1=[0.0, 0.0]
        MR =[0.0, 0.0]
        for i, test_point in enumerate(testData):
            e1, r, gold_e2 = test_point 

            pred_array_1_r_data = np.zeros(opts.num_entities)
            for key in pred_array_1[r].keys():
                pred_array_1_r_data[key] = pred_array_1[r][key]
            pred_array_1_r_seen_e2_i = []
            pred_array_2_e1_seen_e2_i = []
            for key in seen_e2[i]:
                pred_array_1_r_seen_e2_i.append(pred_array_1[r][key])
                pred_array_2_e1_seen_e2_i.append(pred_array_2[e1][key])
            pred_array_1_r_seen_e2_i = np.array(pred_array_1_r_seen_e2_i)
            pred_array_2_e1_seen_e2_i = np.array(pred_array_2_e1_seen_e2_i)

            rank1 = 1 + np.sum(int(pred_array_1[r][gold_e2]) < pred_array_1_r_data) - np.sum(int(pred_array_1[r][gold_e2]) < pred_array_1_r_seen_e2_i)
            same = np.sum(pred_array_1[r][gold_e2] == pred_array_1_r_data) - np.sum(pred_array_1[r][gold_e2] == pred_array_1_r_seen_e2_i)
            assert same >= 0
            assert rank1 >= 0
            rank1 += same/2.0


            pred_array_2_e1_data = np.zeros(opts.num_entities)
            for key in pred_array_2[e1].keys():
                pred_array_2_e1_data[key] = pred_array_2[e1][key]
            rank2 = 1 + np.sum(pred_array_2[e1][gold_e2] < pred_array_2_e1_data) - np.sum(pred_array_2[e1][gold_e2] < pred_array_2_e1_seen_e2_i)
            same = np.sum(pred_array_2[e1][gold_e2] == pred_array_2_e1_data) - np.sum(pred_array_2[e1][gold_e2] == pred_array_2_e1_seen_e2_i)
            assert same >= 0
            assert rank1 >= 0
            rank2 += same/2.0

            MRR[0] += 1.0/rank1
            hits[0] += (rank1 <= 10)
            MR[0] += rank1
            hits1[0] += (rank1 <= 1)
            MRR[1] += 1.0/rank2
            hits[1] += (rank2 <= 10)
            MR[1] += rank2
            hits1[1] += (rank2 <= 1)

        for i in xrange(2):
        	MRR[i] /= len(testData)
        	hits[i] /= len(testData);hits1[i] /= len(testData);MR[i] /= len(testData)

        print("freq(r, e2)\t MRR: %5.4f, HITS@10: %5.4f, HITS@1: %5.4f, MR: %5.4f" %(MRR[0], hits[0],hits1[0],MR[0]))
        print("freq(e1, e2)\t MRR: %5.4f, HITS@10: %5.4f, HITS@1: %5.4f, MR: %5.4f," %(MRR[1], hits[1],hits[1],MR[1]))
        sys.exit(0)

    else:
        print("Please pass a valid model name.")
        sys.exit(1)

    if not (opts.model == "MF-constraint" or opts.model == "MF-constraint-2"):
        aux_model = None
    return aux_model, model, evalFunc
 
 
if __name__ == '__main__':
    opts = get_params() 
    if opts.model == "MF":
        train_entities, train_relations, _ , neg_samples = get_train_data_matrix(opts, verbose=True)
        if opts.oov_train:
            if opts.train:
                trainDataSize = neg_samples.shape[0]
                neg_samples = np.c_[neg_samples, (opts.num_entity_pairs)*np.ones(trainDataSize)]
            opts.neg_samples += 1
    elif opts.model == "MF-constraint":
        train_entities_DM,  train_relations, neg_samples = get_train_data_tensor(opts, verbose=True)
        train_entities, train_relations, _ , neg_samples = get_train_data_matrix(opts, verbose=True)
        if opts.oov_train:
            if opts.train:
                trainDataSize = neg_samples.shape[0]
                neg_samples = np.c_[neg_samples, (opts.num_entity_pairs)*np.ones(trainDataSize)]
            opts.neg_samples += 1
    elif opts.model == "MF-constraint-2":
        train_entities_DM,  train_relations_DM, neg_samples_DM = get_train_data_tensor(opts, verbose=True)
        train_entities, train_relations, _ , neg_samples = get_train_data_matrix(opts, verbose=True)
        if opts.oov_train:
            if opts.train:
                trainDataSize = neg_samples.shape[0]
                neg_samples = np.c_[neg_samples, (opts.num_entity_pairs)*np.ones(trainDataSize)]
                
                neg_column = (opts.num_entities)*np.ones(trainDataSize)
                neg_samples_set1 = np.c_[neg_samples_DM[:, 0: opts.neg_samples], neg_column]
                neg_samples_set2 = np.c_[neg_samples_DM[:, opts.neg_samples:], neg_column]
                neg_samples_DM = np.c_[neg_samples_set1, neg_samples_set2] 

            opts.neg_samples += 1
    else:   
        train_entities,  train_relations, neg_samples = get_train_data_tensor(opts, verbose=True)
        if opts.oov_train:
            if opts.train:
                ''' convert (None, 400) to (None, 402) by appending an OOV after the first 200, then at the end'''
                trainDataSize = neg_samples.shape[0]
                neg_column = (opts.num_entities)*np.ones(trainDataSize)
                neg_samples_set1 = np.c_[neg_samples[:, 0: opts.neg_samples], neg_column]
                neg_samples_set2 = np.c_[neg_samples[:, opts.neg_samples:], neg_column]
                neg_samples = np.c_[neg_samples_set1, neg_samples_set2] 

            opts.neg_samples += 1

    
    aux_model, model, evalFunc = dispatch_model(opts, [train_entities, train_relations, neg_samples])

    
    if (not opts.train):

        if opts.model_path == "":
            model_file = "finalModels/%s" %(opts.model_path)
            if opts.model == "MF-constraint":
                aux_model_file = "MF-constraint-exp/aux_model_%s" %(opts.model_path) 
        else:
            print("Model name missing!");exit()
        print("Using model: %s",model_file)

        print(model.summary())    
            
        model.load_weights("%s" %(model_file), by_name=True)

        if opts.model == "MF-constraint":
            print "Reading from %s" %(aux_model_file)
            aux_model.load_weights("%s" %(aux_model_file), by_name=True)
        mrr, hits = evalFunc(model)
        print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))
    else:
        modelEvalCallback = EvaluateModel(evalFunc, opts, "%s_%d" %(opts.model, opts.oov_train))
        if opts.model == "MF":
            train_MF(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples)
        elif opts.model == "MF-constraint":
            train_MF_constraint(model, aux_model, modelEvalCallback, opts, train_entities_DM, train_entities, train_relations, neg_samples)
        elif opts.model == "MF-constraint-2":
            train_MF_constraint2(model, aux_model, modelEvalCallback, opts, train_entities_DM, train_entities, train_relations, train_relations_DM, neg_samples, neg_samples_DM)

    train_net(model, aux_model,  modelEvalCallback, opts, train_entities, train_relations, neg_samples)
