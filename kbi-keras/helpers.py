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
from time import gmtime, strftime

class UnitEmbeddings(Callback):
    def on_batch_end(self, batch, logs={}):
        
        #entity_embedding_names = set(['entity_type_embeddings', 'relation_head_type_embeddings', 'relation_tail_type_embeddings'])
        entity_embedding_names = set(['entity_embeddings', 'entity_embeddings_E', 'entity_embeddings_DM', 'entity_embeddings_real', 'entity_embeddings_im', 'entity_type_embeddings', 'relation_head_type_embeddings', 'relation_tail_type_embeddings'])
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


class EvaluateModel(Callback):
    def __init__(self, func, opts, model_name = "", aux_model = None, eval_support=True):
        self.func = func
        self.best_mrr  = -1
        self.best_hits = -1
        self.opts = opts
        self.eval_support = eval_support    

        if len(model_name):
            self.model_name = model_name
        else:
            self.model_name = self.opts.model
        
        self.aux_model = aux_model

    def _eval(self):
        print("\n"+str(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        mrr, hits = self.func(self.model)
        print("mrr: %5.4f\thits: %5.4f.\n" %(mrr,hits))
        print("\n"+str(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

        all_embedding_MF = set(['entity_embeddings', 'entity_embeddings_E', 'entity_embeddings_DM', 'relation_embeddings','relation_embeddings_DM','relation_embeddings_MF', 'entity_embeddings_MF']) 
        all_embedding_DM = set(['entity_embeddings', 'entity_embeddings_DM','relation_embeddings','relation_embeddings_DM'])
        all_embedding_complex = set(['entity_embeddings_real', 'entity_embeddings_im','relation_embeddings_real','relation_embeddings_im' ])
        l2=0.0
        '''
        f=open("");train_ent=f.readlines();f.close()
        f=open("");train_ep=f.readlines();f.close()
        train_ent = [ele.strip("\n").split("\t") for ele in train_ent]
        train_e = [int(ele[0]) for ele in train_ent]
        train_e +=[int(ele[1]) for ele in train_ent]
        train_ep =  [int(ele.strip("\n").split("\t")[1]) for ele in train_ep]
        train_e=list(set(train_e))
        train_ep=list(set(train_ep))
        del train_ent'''
        for layer in self.model.layers:
            if layer.name in all_embedding_MF:
                weights = layer.get_weights()[0]
                l2+= np.mean(weights)
            
        print "MF L2 norm at this point:: ", l2
        l2=0.0
        for layer in self.model.layers:
            if layer.name in all_embedding_complex:
                weights = layer.get_weights()[0]
                l2+= np.mean(weights)
        print "Complex L2 norm at this point:: ", l2
        if (mrr > self.best_mrr):
            self.best_mrr = mrr
            self.best_hits = hits
            print("FOUND BEST!!!!!!!!!!!!!!!!!!!!")
            self._save()

    def _save(self):
            save_file = "2_%s_%s_%s_%s_%s_%s_%s_%s_%s.h5" %(self.opts.model,self.opts.dataset,self.opts.vect_dim,self.opts.loss,self.opts.theo_reg,self.opts.add_loss,self.opts.unit_norm_reg,self.opts.oov_train,self.opts.log_file)
            #save_file = "3Feb1_%s_%s_%s.h5" %(self.model_name, self.opts.dataset, self.opts.unit_norm_reg)     
            save_file = save_file.replace("/", "_")    
            save_file = save_file.replace(" ", "") 
            print "Writing to bestModels/%s" %(save_file)
            self.model.save_weights("bestModels/%s" %(save_file), overwrite=True)
            if 0:#self.aux_model:
                if self.opts.model == "MF-constraint-2":
                    x,y = self.aux_model
                    x.save_weights("bestModels/aux_model_%s" %(save_file), overwrite=True)
                    y.save_weights("bestModels/aux_model_DM_%s" %(save_file), overwrite=True)
                else:
                    print "Writing to bestModels/aux_model_%s_%s" %(save_file)
                    self.aux_model.save_weights("bestModels/aux_model_%s_%s" %(save_file), overwrite=True)


    def on_epoch_end(self, epoch, log={} ):
        if (epoch >= self.opts.eval_after and epoch % (self.opts.eval_every) == 0):
            if self.eval_support:
                self._eval()
            else:
                self.func(self.model)
                self._save()    

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

                        
        
class typeNetTrainerCallback(Callback):
    def __init__(self, aux_model, entity_types, entity_neg_types, opts):
        self.aux_model = aux_model
        self.type_pairs = entity_types 
        self.neg_type_pairs = entity_neg_types
        self.idx = np.arange(entity_types.shape[0])
        self.opts = opts
        

    def on_batch_end(self, batch, log= {}):
        np.random.shuffle(self.idx)
        _batch = self.idx[0:self.opts.batch_size]
        (self.aux_model).train_on_batch([ (self.type_pairs)[_batch], (self.neg_type_pairs)[_batch]], np.zeros(self.opts.batch_size)[_batch])

class constraintTrainerCallback(Callback):
    def __init__(self, aux_model, kb_entity_pairs, kb_entities_1, kb_entities_2, opts):
        self.aux_model = aux_model
        self.entity_pairs = kb_entity_pairs 
        self.entities1 = kb_entities_1
        self.entities2 = kb_entities_2
        self.opts = opts
        self.idx = np.arange(kb_entity_pairs.shape[0])

    def on_batch_end(self, batch, log= {}):
        np.random.shuffle(self.idx)
        _batch = self.idx[0:self.opts.batch_size]
        x = np.zeros(self.opts.batch_size)
        (self.aux_model).train_on_batch([ (self.entity_pairs)[_batch], (self.entities1)[_batch], (self.entities2)[_batch]], np.zeros(self.opts.batch_size))

class constraint2TrainerCallback(Callback):
    def __init__(self, aux_model_DM, kb_entities_1, kb_entities_2, kb_relations_DM, neg_samples_DM, opts):
        self.aux_model = aux_model_DM
        self.relations = kb_relations_DM 
        self.entities1 = kb_entities_1
        self.entities2 = kb_entities_2
        self.neg_samples = neg_samples_DM
        self.opts = opts
        self.idx = np.arange(kb_entities_1.shape[0])

    def on_batch_end(self, batch, log= {}):
        np.random.shuffle(self.idx)
        _batch = self.idx[0:self.opts.batch_size]
        x = np.zeros(self.opts.batch_size)
        (self.aux_model).train_on_batch([(self.entities1)[_batch], (self.entities2)[_batch], (self.relations)[_batch], (self.neg_samples)[_batch]], np.zeros(self.opts.batch_size))

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


def has_valid(opts):
    return True     

def get_train_data_tensor(opts, verbose=False):
    data_path = opts.dataset
    if 1:#try:
        folder = "train_dev_tensor" if opts.evalDev and not has_valid(opts) else "train_tensor" 

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
    '''
    except IOError:
        print('invalid dataset path provided.')
        sys.exit(1)
    '''
    return train_entities, train_relations, neg_samples


def get_negative_samples_joint(opts, verbose=False):
    data_path = opts.dataset
    try:
        folder = "train_dev_tensor" if opts.evalDev and not has_valid(opts) else "train_tensor" 
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
    if 1:#try:
        folder = "train_dev_matrix" if opts.evalDev and not has_valid(opts) else "train_matrix" 
        folder_tensor = "train_dev_tensor" if opts.evalDev and not has_valid(opts) else "train_tensor" 
        train_entities = joblib.load("%s/%s/train_pair_entities.joblib" %(data_path, folder))
        train_relations = joblib.load("%s/%s/train_pair_relations.joblib" %(data_path, folder))
        train_pair_ids  = joblib.load("%s/%s/train_entityPairIds.joblib" %(data_path, folder))
        if opts.train:
            # The choice of negative sampling matters for MF.
#            if opts.oov_train:
#                train_neg_samples = joblib.load("%s/%s/train_neg_samples_joint_mf.joblib" %(data_path, folder_tensor))
#            else:
#                train_neg_samples = joblib.load("%s/%s/train_pair_neg_samples.joblib" %(data_path, folder))
            
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
    '''
    except IOError:
        print('invalid dataset path provided.')
        sys.exit(1)
    '''
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
    if 1:#try:
        f = open(opts.dataset + "/entity_pair_id.txt"); entity_pair = f.readlines(); f.close()
     
        entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
        entity_map = {int(ele[2]): (int(ele[0]),int(ele[1])) for ele in entity_pair}
        inverse_map = {(int(ele[0]),int(ele[1])): int(ele[2]) for ele in entity_pair}
        e1s = set(entity_map[ele][0] for ele in train_entities)
        e2s = set(entity_map[ele][1] for ele in train_entities)
    #except IOError:
    #    print("invalid dataset.")
    #    sys.exit(1)

    return entity_map, inverse_map, e1s, e2s

def get_test_data_DM(opts, verbose=False):
    data_path = opts.dataset
    if 1:#try:
        if opts.evalDev:
            folder = "valid_tensor" if has_valid(opts) else "dev_tensor"
        else:
            folder = "test_tensor" 
        print "Reading from %s" %folder
        print opts.evalDev
        test_data = joblib.load('%s/%s/test.joblib' %(data_path, folder))
        oov_flags_1 = joblib.load('%s/%s/oov_flags_e1.joblib' %(data_path, folder))
        oov_flags_2 = joblib.load('%s/%s/oov_flags_e2.joblib' %(data_path, folder))
        oov_e2 = joblib.load('%s/%s/oov_count_e2.joblib' %(data_path, folder))
        #filter_e2 = joblib.load('%s/%s/allowed_baseline.joblib' %(data_path, folder))
        filter_e2 = joblib.load('%s/%s/filter_e2.joblib' %(data_path, folder))
        update_filter = 1#1 ##IMP for normal complex
        if update_filter and not opts.evalDev:
            print "Updating the filter!"
            f=open('%s/encode_train.txt' %(data_path));train=f.readlines();f.close()
            f=open('%s/encode_valid.txt' %(data_path));valid=f.readlines();f.close()
            train = np.array([ele.strip("\n").split("\t") for ele in train],dtype="int")
            valid = np.array([ele.strip("\n").split("\t") for ele in valid],dtype="int")
            e_r = ddict(set)
            for e1,r,e2 in list(train)+list(test_data)+list(valid):
                e_r[(e1,r)].add(e2)
            for i in xrange(filter_e2.shape[0]):
                filter_e2[i] = list(e_r[(test_data[i][0],test_data[i][1])])
            filter_e2 = np.array(filter_e2) 
        if verbose:
            print("DM test data")
            print("oov flags_1",oov_flags_1.shape)
            print("oov flags_2",oov_flags_2.shape)
            print("oov_e2",oov_e2.shape)
            print("filter_e2",filter_e2.shape)
            print("test_data",test_data.shape)
    '''
    except IOError:
        print("invalid dataset.")
        sys.exit(1)
    '''
    return test_data, oov_flags_1, oov_flags_2, filter_e2, oov_e2

def get_path_data_MF(opts, verbose=False):
    data_path = "/home/prachij/code/joint_embedding/code_valid/hypothesis-testing"
    if 1:#try:
        all_data = joblib.load('%s/path-data.joblib' %(data_path))
        np.random.shuffle(train_data)
        train_fraction = int(all_data.shape[0] * 0.7) #70% train rest test
        train_data = all_data[:train_fraction,:]
        test_data  = all_data[train_fraction:,:]
        x_train          = train_data[:,:400]
        y_train          = train_data[:,400:]
        x_test           = test_data[:,:400]
        y_test           = test_data[:,400:]
        assert all_data.shape[1] == x_train.shape[1]+y_train.shape[1]
    '''
    except IOError:
        print("invalid dataset.")
        sys.exit(1)
    '''
    return x_train,y_train,x_test,y_test


def get_test_data_tensor_loss_analysis(opts, verbose=False):
    test_data, oov_flags_1, oov_flags_2, filter_e2, oov_e2 = get_test_data_DM(opts)
    data_path = opts.dataset
    if opts.evalDev:
        type_n = "valid"; folder = "valid_tensor"
    else:
        type_n = "test"; folder = "test_tensor"
    neg_samples = joblib.load("%s/%s/%s_neg_samples.joblib" %(data_path, folder, type_n))
    neg_samples = np.delete(neg_samples,np.concatenate([np.arange(opts.neg_samples,neg_samples.shape[1]/2),np.arange((neg_samples.shape[1]/2)+opts.neg_samples, neg_samples.shape[1])]),axis=1)

    test_entities = np.stack([test_data[:,0],test_data[:,2]]);test_entities=test_entities.transpose()

    test_relations = test_data[:,1] 

    return test_entities, test_relations, neg_samples

def get_aux_data_E(opts):
    data_path = opts.dataset
    if 1:#try:
        oov_flags = joblib.load('%s/oov_flags.joblib' %data_path)
        oov_e2 = joblib.load('%s/oov_e2.joblib' %data_path)
        seen_e2 = joblib.load('%s/seen_e2.joblib' %data_path)
    '''
    except IOError:
        print("invalid dataset.")
        sys.exit(1)
    '''
    return oov_flags, seen_e2, oov_e2

def get_mapping_data_matrix(opts, verbose=False):
    data_path = opts.dataset
    if 1:#try:
        if has_valid(opts):
            folder = "valid_matrix" if opts.evalDev else "test_matrix"
        else:
            folder = "dev_matrix" if opts.evalDev else "test_matrix" 

        mapping_ep_e2 = joblib.load("%s/%s/mapping_ep_e2.joblib" %(data_path, folder))
        
        if verbose:
            print("MF test data")
            print("mapping_ep_e2",mapping_ep_e2.shape)

    #except IOError:
    #    print("invalid dataset.")
    #    sys.exit(1)

    return np.array(mapping_ep_e2)


def get_test_data_matrix(opts, verbose=False):
    data_path = opts.dataset
    if 1:#try:
        if has_valid(opts):
            folder = "valid_matrix" if opts.evalDev else "test_matrix"
        else:
            folder = "dev_matrix" if opts.evalDev else "test_matrix" 

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
    '''
    except IOError:
        print("invalid dataset.")
        sys.exit(1)
    '''
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
    if 1:#try:
        f = open("%s/encode_train.txt" %data_path); training_data = map(lambda line: line.strip().split("\t"), f.readlines()); f.close();
        if os.path.exists("%s/encode_text.txt" %data_path):
            f = open("%s/encode_text.txt" %data_path); training_data += map(lambda line: line.strip().split("\t"), f.readlines()); f.close();

        f = open("%s/encode_test.txt" %data_path); test_data = map(lambda line: line.strip().split("\t"), f.readlines()); f.close();

        train_frequency_features = get_features_helper(training_data)
        test_frequency_features = get_features_helper(test_data)

        return train_frequency_features, test_frequency_features
    '''
    except IOError:
        print("invalid dataset.")
        sys.exit(1)
    '''

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

def update_with_new_embedding_data(testData, old_id, new_e1e2_epId_map, seen_e2, oov_counts, oov_flags,opts):
    '''
    This function updates seen_e2, oov_counts, oov_flags as new entityPair embeddings are added
    '''
    data_path = opts.dataset
    f = open("%s/entity_pair_id.txt" %data_path); entity_pair = f.readlines(); f.close()
    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    ep_e_map = {str(ele[2]): (str(ele[0]),str(ele[1])) for ele in entity_pair}
    e_ep_map = {(str(ele[0]),str(ele[1])): str(ele[2]) for ele in entity_pair}
    old_id   = [str(ele) for ele in old_id]
    e1_e2_map = ddict(set)
    e1_ep_map = ddict(set)
    for ep in old_id:
        e1,e2 = ep_e_map[str(ep)]
        e1_e2_map[str(e1)].add(str(e2))
        e1_ep_map[str(e1)].add(str(ep))
    for key in new_e1e2_epId_map.keys():
        e1_e2_map[str(key[0])].add(str(key[1]))
        e1_ep_map[str(key[0])].add(str(new_e1e2_epId_map[key]))
    print "Seen test ep before update %d" %np.sum(oov_flags)
    old_id = set(old_id)
    k=0;p=0
    for i in xrange(len(testData)):
        e1,r,e2     = testData[i]
        e1 = str(e1);e2=str(e2);r=str(r)
        if len(e1_e2_map[e1]) > 0:
            old_seen    = seen_e2[i]
            old_seen_l  = len(old_seen)
            old_seen   += [int(ele) for ele in list(e1_ep_map[e1])]
            seen_e2[i]  = list(set(old_seen)) 
            
            assert old_seen_l <= len(seen_e2[i])
            oov_counts[i] -= len(seen_e2[i]) - old_seen_l
            
            if not(oov_flags[i]) and (str(e_ep_map[(e1,e2)]) in old_id):#e1_e2_map[e1]:
                oov_flags[i] = 1;p+=1
        if str(e_ep_map[(e1,e2)]) in old_id:
            k+=1
    print "Seen test ep after update %d" %np.sum(oov_flags)
    print "Data for evaluating MF model updated!!!!"
    print k 
    return seen_e2, oov_counts, oov_flags

def get_mappings(data_path):
    f = open("%s/entity_pair_id.txt" %data_path); entity_pair = f.readlines(); f.close()
    f = open("%s/entity_id.txt" %data_path); text_entityId = f.readlines(); f.close()

    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    ep_e_map = {str(ele[2]): (str(ele[0]),str(ele[1])) for ele in entity_pair}
    e_ep_map = {(str(ele[0]),str(ele[1])): str(ele[2]) for ele in entity_pair}

    text_entityId = map(lambda line: line.strip().split("\t"), text_entityId)
    text_e_map = {ele[0]: ele[1] for ele in text_entityId}
    e_text_map = {ele[1]: ele[0] for ele in text_entityId}
    return ep_e_map, e_ep_map, text_e_map, e_text_map

def get_new_embeddings_data(opts):
    data_path = opts.dataset
    f = open("%s/OOV_entity_pair_embeddings.txt" %data_path); oov_entity_pair_data = f.readlines(); f.close()##this file should be of form e1_text\te2_text\te1_old_text\te2_old_text #first 2 are oov ep and next 2 is the ep whose embedding is used OR file format e1_id\te2_id\tnew_embedding
    print "File read: %s/OOV_entity_pair_embeddings.txt" %data_path
    oov_entity_pair_data = map(lambda line: line.strip().split("\t"), oov_entity_pair_data)    
    
    ep_e_map, e_ep_map, text_e_map, e_text_map = get_mappings(data_path)
    
    last_assigned_epId = len(ep_e_map)
    old_id =[]; old_id_new_embeddings=[]; new_id_new_embeddings=[]; new_e1e2_epId_map=ddict(int)
    known_ep_set = set(e_ep_map.keys())
    for data in oov_entity_pair_data:
        if len(data)>3:
            e1_old_text, e2_old_text, e1_new_text, e2_new_text = data ##
        else:
            e1_old_text, e2_old_text, new_embedding = data ##

        try: 
            _ = int(e1_old_text)
            e1_old_id = e1_old_text; e2_old_id = e2_old_text; e1_new_id = e1_new_text; e2_new_id = e2_new_text;
        except:
            e1_old_id = text_e_map[e1_old_text]; e2_old_id = text_e_map[e2_old_text]; e1_new_id = text_e_map[e1_new_text]; e2_new_id = text_e_map[e2_new_text];
        
        if len(data)>3:
            new_embedding_id = int(e_ep_map[(e1_new_id, e2_new_id)])##
        else:
            new_embedding_id = new_embedding

        if (e1_old_id, e2_old_id) in known_ep_set:#if the id to oov ep is already assigned - use that else assign new id
            old_id.append(int(e_ep_map[(e1_old_id, e2_old_id)]))
            old_id_new_embeddings.append(new_embedding_id)
        else:    
            new_id_new_embeddings.append(new_embedding_id)
            new_e1e2_epId_map[(e1_old_id, e2_old_id)] = str(last_assigned_epId)
            last_assigned_epId += 1
    assert last_assigned_epId == len(new_e1e2_epId_map) + len(e_ep_map)
    print "New Embedding extracted!!!"
    return old_id, old_id_new_embeddings, new_id_new_embeddings, new_e1e2_epId_map
        

