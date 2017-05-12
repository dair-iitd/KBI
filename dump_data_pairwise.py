import argparse
import numpy as np
from lib.func import *
import cPickle
import timeit
import joblib
from collections import defaultdict
import sys
import os

'''
This script prepares the data for Matrix Factorization model. To Run:
python dump_data.py -neg_samples [] -dataset [currently supports fb15k-237/WN18/fb15k] -num_entities [for negative sampling]

input:
encode_pair_train.txt, encode_pair_text.txt (empty for WN18/FB15k), encode_pair_test.txt, encode_pair_valid.txt, entity_pair_id.txt
output:
in the directory [named the same as the dataset]:
1) for TRAIN: train_pair_entities.joblib / train_pair_relations.joblib / train_pair_neg_samples.joblib all numpy arrays with same number of rows
2) for TEST: test_pair.joblib/ test_pair_filters.joblib (list of array containing seen entity pairs for the given test relation)
             / test_pair_allowed_data.joblib (array of allowed entity pair for a given test relation, each row is padded with x)
             / test_pair_allowed_index.joblib (list of number of legitimate (non-x) entries in test_pair_allowed_data.joblib)
 
NOTE: FB15k-237 treats every relation as an atom
'''


'''SOME GENERIC UTILITIES'''

def get_params():
    parser = argparse.ArgumentParser(description ='generating negative samples')
    parser.add_argument('-neg_samples', action="store", default=200, dest="neg_samples", type=int)
    parser.add_argument('-num_entity_pairs', action="store", default="14541", dest="num_entity_pairs", type=int)
    parser.add_argument('-num_entities', action="store", default="14541", dest="num_entities", type=int)
    parser.add_argument('-path', action="store", default="", dest="path", type=str)
    parser.add_argument('-random', action="store", default=1, dest="random", type=int)
    
    parser.add_argument('-test_folder', action="store", default="test_matrix", dest="test_folder", type=str)
    parser.add_argument('-train_folder', action="store", default="train_matrix", dest="train_folder", type=str)
    parser.add_argument('-dev_folder', action="store", default="dev_matrix", dest="dev_folder", type=str)

    parser.add_argument('-test_on_valid', action="store", default=0, dest="test_on_valid", type=int)

    opts = parser.parse_args(sys.argv[1:])

    print "neg_samples to be generated: %d" %(opts.neg_samples)
    print "dataset to be used: %s" %(opts.path)#+opts.dataset)
    print "possible object position entities: %d" %(opts.num_entities)
    print "number of entities for creating filters: %d" %(opts.num_entity_pairs)
    return opts

def create_filters(opts, test_data, train_data, valid_data = None):    
    num_entity_pairs = opts.num_entity_pairs
    num_entities = opts.num_entities
    lenData      = len(test_data)
    oov_flags, filters, oov_counts, test_data_modified = ([], [], [], []) 
    all_data = np.concatenate([train_data, test_data])
    if valid_data is not None:
        print("Filtering out validation data")
        all_data = np.concatenate([all_data, valid_data])


    all_data = set((fact[0], fact[1]) for fact in all_data)
    trained_entity_pairs = set([e_pair for (r, e_pair) in train_data])
    f = open(opts.path + "/entity_pair_id.txt"); entity_pair = f.readlines(); f.close()
    # entity_pair only contains a mapping for pairs seen in either train or test
    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    entity_map = {int(ele[2]): (int(ele[0]),int(ele[1])) for ele in entity_pair}
    inverse_map ={(int(ele[0]),int(ele[1])) : int(ele[2]) for ele in entity_pair}  

    for i,test_point in enumerate(test_data):
        # filtered_e2 contains all the legitimate e2 for a given e1 (ofcourse excluding oovs)
        filtered_e2 = []

        #oov_cnt contains all the legitimate e2s such that (e1,e2) forms an oov entity pair
        oov_cnt=0
        r, e_pair= test_point
        e1,e2 = entity_map[e_pair]
        oov_flags.append((e_pair not in trained_entity_pairs))
        test_data_modified.append((r, e_pair, e1, e2))


        for candidate in xrange(num_entities):
            # there is an entity pair corresponding to this in either train or test
            if (e1, candidate) in inverse_map:
                e_pair_idx = inverse_map[(e1, candidate)] 
                candidate_fact = r, e_pair_idx
                found = candidate_fact in all_data # if this is true, we must not compare with this candidate (RULE of filtered measures)
                oov_candidate = e_pair_idx not in trained_entity_pairs # if this is true, we do not have a trained (e1,candidate) embedding
                if not found:
                    if oov_candidate:
                        oov_cnt+=1
                    else:
                        filtered_e2.append(e_pair_idx)

            # there is no entity pair corresponding to this, and hence we must increase the oov_cnt
            else:
                oov_cnt +=1

        filters.append(filtered_e2)
        oov_counts.append(oov_cnt)
        print "\r>> Done with %d/%d test examples: filter ratio= %d"%(i+1, len(test_data), len(filtered_e2)),
        sys.stdout.flush()

    filters = np.array(filters)
    # Important to prevent int32 vs int64 errors.
    if (len(filters.shape) == 2):
        filters = filters.astype('int32')

    return  np.array(test_data_modified ,dtype='int32'), np.array(oov_flags, dtype='int32'), filters, np.array(oov_counts)

''' Routines for creating data joblibs '''
def get_data(data_path_1):
    f=open(data_path_1);train=f.readlines();f.close()
    train_data=[]
    relations=[]
    entities=[]
    for ele in train:
        L=ele.strip().split("\t")
        L=[int(ele) for ele in L]
        train_data.append(L);relations.append(L[0]);entities.append(L[1])
    return np.array(train_data, dtype='int32'), np.array(entities, dtype='int32'), np.array(relations, dtype='int32')


def dump_train_data(opts):
    dataPath = opts.path
    folderName = opts.train_folder
    
    train_data, entityPairs, relations = get_data(dataPath+"/encode_pair_train.txt")

    if os.path.exists(dataPath+"/encode_pair_text.txt"):
        text_data, textPairs, textRelations = get_data(dataPath+"/encode_pair_text.txt")
        train_data = np.concatenate([train_data, text_data])
        entityPairs = np.concatenate([entityPairs, textPairs])
        relations = np.concatenate([relations, textRelations])

    f = open(opts.path + "/entity_pair_id.txt"); entity_pair = f.readlines(); f.close()
    # entity_pair only contains a mapping for pairs seen in either train or test
    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    entity_map = {int(ele[2]): (int(ele[0]),int(ele[1])) for ele in entity_pair}
    entityPairIds = np.array([e for ep in entityPairs for e in entity_map[int(ep)]])
    
    print train_data.shape; 
    print entityPairs.shape; 
    print relations.shape; 

    neg_samples = opts.neg_samples
    
    currPath = os.getcwd()
    if not os.path.exists(dataPath+"/"+folderName):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
    os.chdir(currPath)

    dataPath += "/"+folderName
    
    joblib.dump(entityPairIds, dataPath+"/train_entityPairIds.joblib")
    joblib.dump(entityPairs, dataPath+"/train_pair_entities.joblib")
    joblib.dump(relations, dataPath+"/train_pair_relations.joblib")


    total_entities = opts.num_entity_pairs #this includes ALL the entities-> TRAIN + VALID + TEST

    if opts.random:
        neg_data = np.array(gen_neg_pair_samples_random(train_data, neg_samples, total_entities), dtype='int32')    
    else:
        neg_data = np.array(gen_neg_pair_samples_randomUpdate(train_data, neg_samples, total_entities), dtype='int32')
        
    joblib.dump(neg_data, dataPath+"/train_pair_neg_samples.joblib")
    
def dump_test_data(opts):
    dataPath = opts.path#+opts.dataset
    folderName = opts.test_folder#"test_matrix"
    if opts.test_on_valid:
        print "Encoding Vaidation data"
        test_data, _, _ = get_data(dataPath+"/encode_pair_valid.txt")
        valid_data, _, _ = get_data(dataPath+ "/encode_pair_test.txt")

    else:
        test_data, _, _ = get_data(dataPath+"/encode_pair_test.txt")
        valid_data, _, _ = get_data(dataPath+ "/encode_pair_valid.txt")

    
    train_data, _,_ = get_data(dataPath+"/encode_pair_train.txt")

    if os.path.exists(dataPath+"/encode_pair_text.txt"):
        text_data, textPairs, textRelations = get_data(dataPath+"/encode_pair_text.txt")
        train_data = np.concatenate([train_data, text_data])


    currPath = os.getcwd()
    if not os.path.exists(dataPath+"/"+folderName):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
    os.chdir(currPath)
    
    dataPath += "/"+folderName

    joblib.dump(test_data, dataPath+"/test_pair.joblib")
    test_data_modified, oov_flags, filters, oov_counts = create_filters(opts, test_data, train_data, valid_data)
    joblib.dump(oov_flags, dataPath + "/oov_flags_MF.joblib")
    joblib.dump(filters, dataPath + "/filters_MF.joblib")
    joblib.dump(oov_counts, dataPath + "/oov_counts_MF.joblib")


# 10% of original training data is such that 90% is train remaining and 10% is dev 
def dump_dev_data(opts):
    folderName = opts.dev_folder
    dataPath = opts.path

    train_data, _,_ = get_data(dataPath+"/encode_pair_train.txt")
    len_kb = len(train_data)
    split_idx = int(0.1*len_kb)    
    if os.path.exists(dataPath+"/encode_pair_text.txt"):
        text_data,_,_  = get_data(dataPath+"/encode_pair_text.txt")
        train_data = np.concatenate([train_data, text_data])

    dev_data = train_data[:split_idx]
    train_data = train_data[split_idx:]

    _, oov_flags, filters, oov_counts = create_filters(opts, dev_data, train_data)

    currPath = os.getcwd()
    devPath = "%s/%s" %(dataPath, folderName)
    if not os.path.exists(devPath):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
        os.chdir(currPath)

    
    joblib.dump(dev_data, devPath +"/test_pair.joblib")
    joblib.dump(oov_flags, devPath+"/oov_flags_MF.joblib")
    joblib.dump(filters, devPath+"/filters_MF.joblib")
    joblib.dump(oov_counts, devPath+"/oov_counts_MF.joblib")

    return split_idx


def create_dev_split_from_train(opts):
    split_idx = dump_dev_data(opts)
    dataPath = opts.path
    folderName = opts.train_folder
    train_dev_folder = "train_dev_matrix"

    # get the training data from here
    # 0-split_idx becomes the new training data
    folder_path = "%s/%s" %(dataPath,folderName)
    entityPairs  = joblib.load(folder_path+"/train_pair_entities.joblib")[split_idx:]
    relations = joblib.load(folder_path+"/train_pair_relations.joblib")[split_idx:]
    entityPairIds = joblib.load(folder_path+"/train_entityPairIds.joblib")[split_idx:]
    neg_data = joblib.load(folder_path+"/train_pair_neg_samples.joblib")[split_idx:]


    # check if the train-dev folder exists inside dataPath. if not, travel to dataPath and mkdir it.
    # also don't forget to chdir back to the original path

    dev_folder_path = "%s/%s" %(dataPath, train_dev_folder)
    currPath = os.getcwd()
    if not os.path.exists(dev_folder_path):
        os.chdir(dataPath)
        os.system("mkdir "+ train_dev_folder)
        os.chdir(currPath)

    
    joblib.dump(entityPairs, dev_folder_path+"/train_pair_entities.joblib")
    joblib.dump(relations, dev_folder_path+"/train_pair_relations.joblib")        
    joblib.dump(neg_data, dev_folder_path+"/train_pair_neg_samples.joblib")
    joblib.dump(entityPairIds, dev_folder_path+"/train_entityPairIds.joblib")[split_idx:]



if __name__ == '__main__':
    opts = get_params()
    #dump_train_data(opts)
    dump_test_data(opts)
    #create_dev_split_from_train(opts)
