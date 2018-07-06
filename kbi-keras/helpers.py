from keras.layers import *
from lib.wrappers import *
from lib.MFmodel import *
from lib.new_eval import *
from keras.callbacks import *
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.generic_utils import Progbar
import theano
import theano.tensor as T
import numpy as np
import joblib,re
import cPickle
import os,sys
import resource
import h5py
import scipy.spatial.distance as dist
from collections import defaultdict as ddict
import random


# == keras callback for setting norm of entity embeddings to 1 after every gradient update. Helps performance. Normalising layers are for calibrating weights of DM/MF in a joint model.

class UnitEmbeddings(Callback):
    def on_batch_end(self, batch, logs={}):
        entity_embedding_names = set(['entity_embeddings', 'entity_embeddings_E', 'entity_embeddings_DM', 'entity_embeddings_real', 'entity_embeddings_im'])
        normalizing_layers = set(['normalize_MF', 'normalize_DM'])

        for layer in self.model.layers:
            if layer.name in entity_embedding_names:
                entity_weights = layer.get_weights()[0]
                entity_weights /= np.linalg.norm(entity_weights, axis=1, keepdims=True)
                layer.set_weights([entity_weights])

            elif layer.name in normalizing_layers:
                curr_W, curr_b = layer.get_weights()
                curr_W = np.clip(curr_W, 0.0001, 0.1)
                curr_b = np.clip(curr_b, -0.005, 0.005)
                layer.set_weights([curr_W, curr_b])

# == callback function for evaluting the model after every opts.eval_every epoch. Implements checkpointing and logging when the best model is found.

class EvaluateModel(Callback):
    def __init__(self, func, opts, model_name = "",  eval_support=True):
        self.func = func
        self.best_mrr  = -1
        self.best_hits = -1
        self.opts = opts
        self.eval_support = eval_support    

        if len(model_name):
            self.model_name = model_name
        else:
            self.model_name = self.opts.model
         

    def _eval(self):
        mrr, hits = self.func(self.model)
        print("mrr: %5.4f\thits: %5.4f.\n" %(mrr,hits))
        if (mrr > self.best_mrr):
            self.best_mrr = mrr
            self.best_hits = hits
            print("FOUND BEST!!!!!!!!!!!!!!!!!!!!")
            self._save()

    def _save(self):
            save_file = "%s_%s.h5" %(self.model_name, self.opts.dataset)     
            save_file = save_file.replace("/", "_")    
            #print "Writing to bestModels/%s" %(save_file)
            self.model.save_weights("bestModels/%s" %(save_file), overwrite=True)


    def on_epoch_end(self, epoch, log={} ):
        if (epoch >= self.opts.eval_after and epoch % (self.opts.eval_every) == 0):
            if self.eval_support:
                self._eval()
            else:
                self.func(self.model)
                self._save()    


# == Dropout callback for preventing overfitting. Deprecated
class Dropout_e(Callback):
    def __init__(self, opts):
        self.opts = opts

    def on_batch_end(self, batch, logs={}):
        entity_embedding_names = set(['entity_embeddings_DM', 'entity_embeddings_MF'])

        for layer in self.model.layers:
            if layer.name in entity_embedding_names:
                if layer.name == "entity_embeddings_DM":
                    bound = self.opts.dropout_DM
                elif layer.name == "entity_embeddings_MF":
                    bound = self.opts.dropout_MF
                if random.random() < bound: 
                    entity_weights = layer.get_weights()[0]
                    entity_weights *= 0.0
                    layer.set_weights([entity_weights])

                        
# == Callback for joint training of an entity typing model along with a KBI model. Alternates between gradient updates of KBI and entity typing.        
class typeNetTrainerCallback(Callback):
    def __init__(self, aux_model, entity_types, entity_neg_types, opts):
        self.aux_model = aux_model
        self.type_pairs = entity_types 
        self.neg_type_pairs = entity_neg_types
        self.idx = np.arange(entity_types.shape[0])
        self.opts = opts
        

    def on_batch_end(self, batch, log= {}):
        np.random.shuffle(self.idx)
        _batch = self.idx[0:opts.batch_size]
        (self.aux_model).train_on_batch([ (self.type_pairs)[_batch], (self.neg_type_pairs)[_batch]], np.zeros(opts.batch_size))


def get_pretrained(modelMF, modelDM):
    for layer in modelMF.layers:
        if layer.name == 'entity_embeddings':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
    
    for layer in modelDM.layers:
        if layer.name == 'entity_embeddings':
            oov_weights = layer.get_weights()[0]
 
    return entity_weights, relation_weights, oov_weights



# == A collection of helper functions for reading Tensor/Matrix factorization data that has been preprocessed, id-fied and stored as numpy joblibs.

def get_train_data_tensor(opts, verbose=False):
    data_path = opts.dataset
    try:
        folder = "train_dev_tensor" if opts.evalDev else "train_tensor" 

        train_entities  = joblib.load('%s/%s/train_entities.joblib' %(data_path, folder))
        train_relations = joblib.load('%s/%s/train_relations.joblib'%(data_path, folder))
        if opts.train:
            neg_samples = joblib.load("%s/%s/train_neg_samples.joblib" %(data_path, folder))
            neg_samples = np.delete(neg_samples,np.concatenate([np.arange(opts.neg_samples,neg_samples.shape[1]/2),np.arange((neg_samples.shape[1]/2)+opts.neg_samples, neg_samples.shape[1])]),axis=1)
        else:
            neg_samples     = np.array([])

        if verbose:
            print("train entities", train_entities.shape)
            print ("train relations",train_relations.shape)
            print ("neg samples",neg_samples.shape)

    except IOError:
        print('invalid dataset path provided.')
        sys.exit(1)

    return train_entities, train_relations, neg_samples


def get_negative_samples_joint(opts, verbose=False):
    data_path = opts.dataset
    try:
        folder = "train_dev_tensor" if opts.evalDev else "train_tensor" 
        if opts.train:
            neg_data_tf = joblib.load("%s/%s/train_neg_samples_joint_tf.joblib" %(data_path, folder))
            neg_data_mf = joblib.load("%s/%s/train_neg_samples_joint_mf.joblib" %(data_path, folder))
            delete_from = opts.neg_samples
            delete_till_tf = neg_data_tf.shape[1]
            delete_till_mf = neg_data_mf.shape[1]

            neg_data_tf = np.delete(neg_data_tf,np.arange(delete_from, delete_till_tf),axis=1)
            neg_data_mf = np.delete(neg_data_mf,np.arange(delete_from, delete_till_mf),axis=1)
        else:
            neg_data_tf = np.array([])
            neg_data_mf = np.array([])

        if verbose:
            print("neg samples mf", neg_data_mf.shape)
            print("neg samples tf", neg_data_tf.shape)

    except IOError:
        print("invalid dataset provided")
        sys.exit(1)

    return neg_data_tf, neg_data_mf        


def get_train_data_matrix(opts, verbose=False):
    data_path = opts.dataset
    try:
        folder = "train_dev_matrix" if opts.evalDev else "train_matrix" 
        folder_tensor = "train_dev_tensor" if opts.evalDev  else "train_tensor" 
        train_entities = joblib.load("%s/%s/train_pair_entities.joblib" %(data_path, folder))
        train_relations = joblib.load("%s/%s/train_pair_relations.joblib" %(data_path, folder))
        train_pair_ids  = joblib.load("%s/%s/train_entityPairIds.joblib" %(data_path, folder))
        if opts.train:
            train_neg_samples = joblib.load("%s/%s/train_pair_neg_samples.joblib" %(data_path, folder))
            delete_from = opts.neg_samples
            delete_till = train_neg_samples.shape[1]
            train_neg_samples = np.delete(train_neg_samples,np.arange(delete_from, delete_till),axis=1)
        else:
            train_neg_samples = np.array([])

        if verbose:
            print("train entities", train_entities.shape)
            print("train relations",train_relations.shape)
            print("train pair ids",train_pair_ids.shape)
            print("train neg samples",train_neg_samples.shape)

    except IOError:
        print('invalid dataset path provided.')
        sys.exit(1)

    return train_entities, train_relations, train_pair_ids, train_neg_samples 

def get_typing_data(opts):
    data_path = opts.dataset
    try:
        entity_types = joblib.load('%s/entity_types.joblib' %data_path)
        if opts.train:
            entity_neg_types = joblib.load('%s/entity_neg_types.joblib' %data_path)
        else:
            entity_neg_types = np.array([])

    except IOError:
        print('typing information not present.')
        sys.exit(1)

    return [entity_types, entity_neg_types]


def get_aux_data(opts, train_entities):
    data_path = opts.dataset
    try:
        f = open(opts.dataset + "/entity_pair_id.txt"); entity_pair = f.readlines(); f.close()
     
        entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
        entity_map = {int(ele[2]): (int(ele[0]),int(ele[1])) for ele in entity_pair}
        inverse_map = {(int(ele[0]),int(ele[1])): int(ele[2]) for ele in entity_pair}
        e1s = set(entity_map[ele][0] for ele in train_entities)
        e2s = set(entity_map[ele][1] for ele in train_entities)
    except IOError:
        print("invalid dataset.")
        sys.exit(1)

    return entity_map, inverse_map, e1s, e2s

def get_test_data_DM(opts, verbose=False):
    data_path = opts.dataset
    try:
        if opts.evalDev:
            folder = "valid_tensor"
        else:
            folder = "test_tensor" 

        test_data = joblib.load('%s/%s/test.joblib' %(data_path, folder))
        oov_flags_1 = joblib.load('%s/%s/oov_flags_e1.joblib' %(data_path, folder))
        oov_flags_2 = joblib.load('%s/%s/oov_flags_e2.joblib' %(data_path, folder))
        oov_e2 = joblib.load('%s/%s/oov_count_e2.joblib' %(data_path, folder))
        filter_e2 = joblib.load('%s/%s/filter_e2.joblib' %(data_path, folder))

        if verbose:
            print("DM test data")
            print("oov flags_1",oov_flags_1.shape)
            print("oov flags_2",oov_flags_2.shape)
            print("oov_e2",oov_e2.shape)
            print("filter_e2",filter_e2.shape)
            print("test_data",test_data.shape)

    except IOError:
        print("invalid dataset.")
        sys.exit(1)

    return test_data, oov_flags_1, oov_flags_2, filter_e2, oov_e2


def get_aux_data_E(opts):
    data_path = opts.dataset
    try:
        oov_flags = joblib.load('%s/oov_flags.joblib' %data_path)
        oov_e2 = joblib.load('%s/oov_e2.joblib' %data_path)
        seen_e2 = joblib.load('%s/seen_e2.joblib' %data_path)
    except IOError:
        print("invalid dataset.")
        sys.exit(1)

    return oov_flags, seen_e2, oov_e2

def get_test_data_matrix(opts, verbose=False):
    data_path = opts.dataset
    try:
        folder = "valid_matrix" if opts.evalDev else "test_matrix"

        oov_flags = joblib.load("%s/%s/oov_flags_MF.joblib" %(data_path, folder))
        seen_e2 = joblib.load("%s/%s/filters_MF.joblib" %(data_path, folder))
        oov_counts = joblib.load("%s/%s/oov_counts_MF.joblib" %(data_path, folder))
        test_data = joblib.load("%s/%s/test_pair.joblib" %(data_path, folder))

        if verbose:
            print("MF test data")
            print("oov flags",oov_flags.shape)
            print("seen_e2",seen_e2.shape)
            print("oov_counts",oov_counts.shape)
            print("test_data",test_data.shape)

    except IOError:
        print("invalid dataset.")
        sys.exit(1)

    return test_data, oov_flags, seen_e2, oov_counts


def embedding_distance(opts, model, verbose=True):

    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_E':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings':
            relation_weights = layer.get_weights()[0]
    
    num_relations = opts.num_relations
    distance_matrix = np.zeros((num_relations, num_relations))    
    
    for i in xrange(num_relations):
        for j in xrange(num_relations):
            distance_matrix[i][j] = dist.cosine(relation_weights[i],relation_weights[j])

    if verbose:
        print '{0:"red", 1:"back", 2:"green", 3:"blue"}'
        index = ['x']+range(num_relations)
        print ("\t".join(map(str,index)))
        print "-"*170
        for i in index[1:]:
            print str(i)+"\t"+("\t").join(map(lambda num: "%5.4f" %num,distance_matrix[i]))
            print "-"*170

    return distance_matrix
    
'''
frequency features
- number of times e1 was seen in train set
- number of entity pairs seen with r
- number of entities seen with r
- number of times (e1, ?) entity pairs seen in train set
'''


def get_features_helper(training_data):    
    e1_counter = ddict(int)
    r_entity_pair_counter = ddict(set)
    r_entity_counter = ddict(set)
    e1_subject_counter = ddict(int)

    for (e1, r, e2) in training_data:
        e1_counter[e1] += 1; e1_counter[e2] += 1
        r_entity_pair_counter[r].add((e1,e2))
        r_entity_counter[r].add(e1); r_entity_counter[r].add(e2)
        e1_subject_counter[e1] += 1

    frequency_feature_matrix = np.zeros((len(training_data), 4))

    for i, data in enumerate(training_data):
        e1, r, e2 = data
        f1 = e1_counter[e1]; f2 = len(r_entity_pair_counter[r]); f3 = len(r_entity_counter[r]); f4 = e1_subject_counter[e1]
        frequency_feature_matrix[i] = [f1, f2, f3, f4]

    return frequency_feature_matrix

def get_other_features(opts, verbose=True):
    data_path = opts.dataset
    try:
        f = open("%s/encode_train.txt" %data_path); training_data = map(lambda line: line.strip().split("\t"), f.readlines()); f.close();
        if os.path.exists("%s/encode_text.txt" %data_path):
            f = open("%s/encode_text.txt" %data_path); training_data += map(lambda line: line.strip().split("\t"), f.readlines()); f.close();

        f = open("%s/encode_test.txt" %data_path); test_data = map(lambda line: line.strip().split("\t"), f.readlines()); f.close();

        train_frequency_features = get_features_helper(training_data)
        test_frequency_features = get_features_helper(test_data)

        return train_frequency_features, test_frequency_features

    except IOError:
        print("invalid dataset.")
        sys.exit(1)


# == partitions candidate e2s for an (e1, r, ?) query into 2 categories based on whether (e1, e2') is OOV or not. We use this to speed up MF-OOV evaluation.
def convertToSets(opts, allowedEP, filter_e2):
    data_path = opts.dataset
    f = open("%s/entity_pair_id.txt" %data_path); entity_pair = f.readlines(); f.close()

    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    entity_map = {int(ele[2]): (int(ele[0]),int(ele[1])) for ele in entity_pair}

    def f(seen_e2_row):
        if seen_e2_row:
            return [entity_map[i][1] for i in seen_e2_row]
        else:
            return []
         
    set1_e2 = map(lambda seen_e2_row: f(seen_e2_row), allowedEP)
    filter_e2 = [set(filters) for filters in filter_e2]

    set2_e2 = map(lambda ele: [i for i in xrange(opts.num_entities) if i not in ele[0] and i not in ele[1]], zip(set1_e2, filter_e2))
 

    return set1_e2, set2_e2



