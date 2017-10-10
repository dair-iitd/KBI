import warnings
warnings.simplefilter("ignore", UserWarning)
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


def train_baselines(opts, inputs):
    train_entities, train_relations, neg_samples = inputs

    print("Running baseline models")
    pred_array_1 = np.zeros((opts.num_relations, opts.num_entities)) 
    pred_array_2 = np.zeros((opts.num_entities,  opts.num_entities)) 
    trainDataSize = train_entities.shape[0]
    testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)
    for i in xrange(trainDataSize):
        e1,e2 = train_entities[i] 
        r = train_relations[i]
        pred_array_1[r][e2] += 1
        pred_array_2[e1][e2] += 1

    MRR=[0.0, 0.0]
    hits=[0.0, 0.0]
    for i, test_point in enumerate(testData):
        e1, r, gold_e2 = test_point 

        rank1 = 1 + np.sum(pred_array_1[r][gold_e2] < pred_array_1[r]) - np.sum(pred_array_1[r][gold_e2] < pred_array_1[r][seen_e2[i]])
        same = np.sum(pred_array_1[r][gold_e2] == pred_array_1[r]) - np.sum(pred_array_1[r][gold_e2] == pred_array_1[r][seen_e2[i]])
        rank1 += same/2.0

        rank2 = 1 + np.sum(pred_array_2[e1][gold_e2] < pred_array_2[e1]) - np.sum(pred_array_2[e1][gold_e2] < pred_array_2[e1][seen_e2[i]])
        same = np.sum(pred_array_2[e1][gold_e2] == pred_array_2[e1]) - np.sum(pred_array_2[e1][gold_e2] == pred_array_2[e1][seen_e2[i]])
        rank2 += same/2.0


        MRR[0] += 1.0/rank1
        hits[0] += (rank1 <= 10)
        MRR[1] += 1.0/rank2
        hits[1] += (rank2 <= 10)

    for i in xrange(2):
        MRR[i] /= len(testData)
        hits[i] /= len(testData)


    return MRR, hits


'''

main training loop for training Tensor factorization models
- train_entities: a |train_data| x 2 array containing (e_1, e_2) values for each training point
- train_relations: a (|train_data|,) array containing r values for each training point
- neg_samples: a |train_data| x (2*opts.neg_samples) where the first (opts.neg_samples) values are (e2') and next (opts.neg_samples) values are e1'
'''
def train_net(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples):
    callback_functions = [modelEvalCallback]

    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    dummy_y = np.zeros((train_entities.shape[0],1))
    history = model.fit(x=[train_entities, train_relations, neg_samples], y=dummy_y, batch_size=opts.batch_size, epochs=opts.nb_epochs, callbacks=callback_functions)

    print("="*50)
    print("best MRR: %5.4f" %modelEvalCallback.best_mrr) 
    print("best hits@10: %5.4f" %modelEvalCallback.best_hits) 
    print("="*50)


'''
main training loop for training Tensor factorization models
- train_entities: a |train_data| x 1 array containing (ep_12) values for each training point
- train_relations: a (|train_data|,) array containing r values for each training point
- neg_samples: a |train_data| x (opts.neg_samples) containing ep_12' of the form (e_1, e_2')

'''

def train_MF(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples):
    callback_functions = [modelEvalCallback]
    if opts.unit_norm_reg:
        print("entity vectors normalized after every update.")
        callback_functions.append(UnitEmbeddings())

    dummy_y = np.zeros(train_entities.shape[0])
    num_entity_pairs  = opts.num_entity_pairs
    trainDataSize = train_entities.shape[0]
    history = model.fit(x=[train_entities, train_relations, neg_samples], y=dummy_y, batch_size=opts.batch_size, nb_epoch=opts.nb_epochs, callbacks=callback_functions)

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
        evalFunc = lambda model: scoreDM_OOV(model,opts, testData, [train_entities, train_relations], 
                                            oov_flags_1, oov_flags_2,seen_e2, oov_e2, score_fn, score_fn_aux, distMult_scorer,oov_flags)                
    
    
    elif opts.model == "E":
        model = build_atomic_model(opts, getE_score)
        score_fn  = E_oovEval() 
        testData, oov_flags_1, _,  seen_e2, oov_e2 = get_test_data_DM(opts)
        evalFunc = lambda model: scoreE_OOV(model,train_entities,  testData, oov_flags_1, seen_e2, oov_e2, score_fn)
         
    elif opts.model == "TransE":
        model = build_atomic_model(opts, getTransE_score)
        testData, oov_flags_1, oov_flags_2, seen_e2, oov_e2 = get_test_data_DM(opts, True)

        score_fn = TransE_oovEval()
        evalFunc = lambda model: scoreTransE_OOV(opts, model,testData, train_entities, oov_flags_1, oov_flags_2,seen_e2, oov_e2, score_fn)        
        
    elif opts.model == "MF":
        model = build_MFModel(opts)
        testData, oov_flags, seen_e2, oov_counts = get_test_data_matrix(opts, verbose=True)

        # if we're not doing oov evaluation, this is essentially what we are saying
        if not opts.oov_eval:
            oov_flags = np.zeros_like(oov_flags)
            oov_counts = np.zeros_like(oov_counts)

        counter = Counter(train_entities)
        evalFunc = lambda model: scoreMF_OOV_naive(model, opts, counter, testData, oov_flags, seen_e2, oov_counts, MF_oovEval_naive()) 



    elif opts.model == "freq":
        MRR, hits = train_baselines(opts, inputs)
        print("freq(r, e2)\t MRR: %5.4f, HITS@10: %5.4f" %(MRR[0], hits[0]))
        print("freq(e1, e2)\t MRR: %5.4f, HITS@10: %5.4f" %(MRR[1], hits[1]))
        sys.exit(0)


    else:
        print("Please pass a valid model name.")
        sys.exit(1)

    return model, evalFunc
 
 
if __name__ == '__main__':
    opts = get_params() 
    if opts.model == "MF":
        train_entities, train_relations, _ , neg_samples = get_train_data_matrix(opts, verbose=True)
        if opts.oov_train:
            if opts.train:
                trainDataSize = neg_samples.shape[0]
                neg_samples = np.c_[neg_samples, (opts.num_entity_pairs)*np.ones(trainDataSize)]
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

    model, evalFunc = dispatch_model(opts, [train_entities, train_relations, neg_samples])

    
    if (not opts.train):

        if opts.model_path == "":
            model_name = "%s_%d" %(opts.model, opts.oov_train)
            model_file = "%s_%s.h5" %(model_name, opts.dataset)
            model_file = "bestModels/%s" %(model_file.replace("/", "_"))
        else:
            model_file = "finalModels/%s" %(opts.model_path)


        print(model.summary())    
        print "Reading from %s" %(model_file)
        model.load_weights("%s" %(model_file), by_name=True)
        mrr, hits = evalFunc(model)

        print("MRR: %5.4f, HITS: %5.4f" %(mrr, hits))


    else:
        modelEvalCallback = EvaluateModel(evalFunc, opts, "%s_%d" %(opts.model, opts.oov_train))
        if opts.model == "MF":
            train_MF(model, modelEvalCallback, opts, train_entities, train_relations, neg_samples)
        else:
            train_net(model,  modelEvalCallback, opts, train_entities, train_relations, neg_samples)
