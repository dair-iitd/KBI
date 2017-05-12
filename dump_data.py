import argparse
import numpy as np
from lib.func import *
from lib.neg_samples_joint import *
import cPickle
import timeit
import joblib
from keras.preprocessing.sequence import pad_sequences
import sys
import os

'''
This script prepares the data for all the models. To Run:
python dump_data.py -neg_samples [] -dataset [currently supports fb15k-237/ nyt] -num_entities [for negative sampling]


output:
in the directory [named the same as the dataset]:
1) for TRAIN: train_entities.joblib / train_relations.joblib / train_neg_samples.joblib all numpy arrays with same number of rows
2) for TEST: test.joblib/ test_filters.joblib
 
NOTE: FB15k-237 treats every relation as an atom, whereas for NYT we have both atomic and non atomic models. For atomic NYT specify nytAtomic
and for non atomic, nytNonAtomic
'''


'''SOME GENERIC UTILITIES'''

def get_params():
    parser = argparse.ArgumentParser(description ='generating negative samples')
    parser.add_argument('-neg_samples', action="store", default=200, dest="neg_samples", type=int)
    #parser.add_argument('-dataset', action="store", default="FB15k-237", dest="dataset", type=str)
    parser.add_argument('-num_entities', action="store", default=14541, dest="num_entities", type=int)
    parser.add_argument('-num_entity_pairs', action="store", default= -1, dest="num_entity_pairs", type=int)
    parser.add_argument('-path', action="store", default="", dest="path", type=str)
    parser.add_argument('-test_folder', action="store", default="test_tensor", dest="test_folder", type=str)
    parser.add_argument('-train_folder', action="store", default="train_tensor", dest="train_folder", type=str)
    parser.add_argument('-dev_folder', action="store", default="dev_tensor", dest="dev_folder", type=str)
    parser.add_argument('-test_on_valid', action="store", default=0, dest="test_on_valid", type=int)
    parser.add_argument('-neg_joint', action="store", default=0, dest="neg_joint", type=int)

    opts = parser.parse_args(sys.argv[1:])
    print "neg_samples to be generated: %d" %(opts.neg_samples)
    print "dataset to be used: %s" %(opts.path)#+opts.dataset)
    print "number of entities for creating filters: %d" %(opts.num_entities)

    if opts.neg_joint != 0 and opts.num_entity_pairs == -1:
        print "please provide num_entity_pairs if you want to generate neg samples for the joint model"
        sys.exit(0)

    return opts

def create_filters(opts, test_data, train_data, valid_data = None):
    ''' train_data and test_data: consist of triplets with 3 numbers per datapoint 
        returns:
            -oov_flags_e1: 0 if e1 is not OOV else 1
            -oov_flags_e2: 0 if e2 is not OOV else 1
            -filter_e2: things to be filtered out (included e2s obtained via filtered measures, and oovs)
            -oov_e2: #e2 which are OOVs given an (e1, r, ?) and such that (e1, r, e2) is not seen in train + test
    '''
    num_entities = opts.num_entities
    lenData      = len(test_data)

    oov_flags_e1,oov_flags_e2, filter_e2, oov_e2 = ([], [], [], []) 


    all_data = np.concatenate([train_data, test_data])
    if valid_data is not None:
        print("Filtering out validation data")
        all_data = np.concatenate([all_data, valid_data])



    all_data = set((fact[0], fact[1], fact[2]) for fact in all_data)
    trained_entities = set([e for (e1, r, e2) in train_data for e in (e1,e2)])

    for i,test_point in enumerate(test_data):
        seen = []
        oov=0
        e1, r, e2 = test_point
        oov_flags_e1.append((e1 not in trained_entities))
        oov_flags_e2.append((e2 not in trained_entities))

        for candidate in xrange(num_entities):
            candidate_triple = e1, r, candidate
            # if a candidate triple is in train or test, we DO NOT evaluate against it, and hence put it in seen.
            # so, for comparing with OOVs we only take those OOVs such that the corresponding triple is not in train/test
            found = candidate_triple in all_data
            oov_candidate = candidate not in trained_entities
            if (not found and oov_candidate):
                oov+=1
            if (found or oov_candidate):
                seen.append(candidate)               

        filter_e2.append(seen)
        oov_e2.append(oov)
        print "\r>> Done with %d/%d test examples: filtered e2s= %d, oov e2s= %d"%(i+1, len(test_data), len(seen), oov),
        sys.stdout.flush()

    filter_e2 = np.array(filter_e2)

    # Important to prevent int32 vs int64 errors.
    if (len(filter_e2.shape) == 2):
        filter_e2 = filter_e2.astype('int32')

    return  np.array(oov_flags_e1, dtype='int32'),np.array(oov_flags_e2, dtype='int32'), filter_e2, np.array(oov_e2, dtype='int32')

def get_data(data_path):
    f=open(data_path);train=f.readlines();f.close()
    train_data=[]
    relations=[]
    entities=[]
    for ele in train:
            L=ele.strip().split("\t")
            L=[int(ele) for ele in L]
            train_data.append(L);relations.append(L[1]);entities.append([L[0], L[2]])
    return np.array(train_data, dtype='int32'), np.array(entities, dtype='int32'), np.array(relations, dtype='int32')

def dump_train_data(opts):
    dataPath = opts.path#+opts.dataset
    folderName = opts.train_folder#"train_tensor"
    train_data, entities, relations = get_data(dataPath+"/encode_train.txt")

    f = open(opts.path + "/entity_pair_id.txt"); entity_pair = f.readlines(); f.close()
    # entity_pair only contains a mapping for pairs seen in either train or test
    entity_pair = map(lambda line: line.strip().split("\t"), entity_pair)
    inverse_map ={(int(ele[0]),int(ele[1])) : int(ele[2]) for ele in entity_pair}  

    if os.path.exists(dataPath+"/encode_text.txt"):
        text_data, text_entities , text_relations  = get_data(dataPath+"/encode_text.txt")
        train_data = np.concatenate([train_data, text_data])
        entities   = np.concatenate([entities, text_entities])
        relations  = np.concatenate([relations, text_relations])

    neg_samples = opts.neg_samples  
    entityPairToId = {}

    for (e1, r, e2) in train_data:
        entityPairToId[(e1, e2)] = inverse_map[(e1,e2)]

    currPath = os.getcwd()
    if not os.path.exists(dataPath+"/"+folderName):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
    os.chdir(currPath)

    dataPath += "/"+folderName
    joblib.dump(entities, dataPath+"/train_entities.joblib")
    joblib.dump(relations, dataPath+"/train_relations.joblib")

    total_entities = opts.num_entities #this includes ALL the entities-> TRAIN + VALID + TEST
    total_entity_pairs = opts.num_entity_pairs
    if opts.neg_joint:
        neg_data_tf, neg_data_mf = gen_neg_joint(train_data, neg_samples, total_entities, total_entity_pairs, entityPairToId)
        joblib.dump(neg_data_tf, dataPath+"/train_neg_samples_joint_tf.joblib")
        joblib.dump(neg_data_mf, dataPath+"/train_neg_samples_joint_mf.joblib")

    else:
        neg_data = np.array(gen_neg_samples(train_data, neg_samples, total_entities), dtype='int32')    
        joblib.dump(neg_data, dataPath+"/train_neg_samples.joblib")



def dump_test_data(opts):
    dataPath = opts.path#+opts.dataset
    folderName = opts.test_folder#"test_tensor"
    if opts.test_on_valid:
        print "Encoding Vaidation data"
        test_data, _, _ = get_data(dataPath+"/encode_valid.txt")
        valid_data, _, _ = get_data(dataPath+ "/encode_test.txt")

    else:
        test_data, _, _ = get_data(dataPath+"/encode_test.txt")
        valid_data, _, _ = get_data(dataPath+ "/encode_valid.txt")

    
    train_data, _,_ = get_data(dataPath+"/encode_train.txt")

    if os.path.exists(dataPath+"/encode_text.txt"):
        text_data,_,_  = get_data(dataPath+"/encode_text.txt")
        train_data = np.concatenate([train_data, text_data])


    oov_flags_e1, oov_flags_e2, filter_e2, oov_e2 = create_filters(opts, test_data, train_data, valid_data)
    
    currPath = os.getcwd()
    if not os.path.exists(dataPath+"/"+folderName):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
    os.chdir(currPath)
    
    dataPath += "/"+folderName
    
    joblib.dump(test_data, dataPath+"/test.joblib")

    joblib.dump(oov_flags_e1, dataPath+"/oov_flags_e1.joblib")
    joblib.dump(oov_flags_e2, dataPath+"/oov_flags_e2.joblib")
    joblib.dump(filter_e2, dataPath+"/filter_e2.joblib")
    joblib.dump(oov_e2, dataPath+"/oov_count_e2.joblib")
    

# 10% of original training data is such that 90% is train remaining and 10% is dev 
def dump_dev_data(opts):
    folderName = opts.dev_folder
    dataPath = opts.path


    train_data, _,_ = get_data(dataPath+"/encode_train.txt")
    split_idx = int(0.1*len(train_data))
    if os.path.exists(dataPath+"/encode_text.txt"):
        text_data,_,_  = get_data(dataPath+"/encode_text.txt")
        train_data = np.concatenate([train_data, text_data])

    print("%d elements out of %d to be used as dev set." %(split_idx, len(train_data)))
    dev_data = train_data[:split_idx]
    train_data_remaining = train_data[split_idx:]

    oov_flags_e1, oov_flags_e2, filter_e2, oov_e2 = create_filters(opts, dev_data, train_data)

    currPath = os.getcwd()
    if not os.path.exists(dataPath+"/"+folderName):
        os.chdir(dataPath)
        os.system("mkdir "+folderName)
    os.chdir(currPath)

    dataPath += "/"+folderName
    
    joblib.dump(dev_data, dataPath+"/test.joblib")

    joblib.dump(oov_flags_e1, dataPath+"/oov_flags_e1.joblib")
    joblib.dump(oov_flags_e2, dataPath+"/oov_flags_e2.joblib")
    joblib.dump(filter_e2, dataPath+"/filter_e2.joblib")

    joblib.dump(oov_e2, dataPath+"/oov_count_e2.joblib")
    return split_idx


def create_dev_split_from_train(opts):
    split_idx = dump_dev_data(opts)
    dataPath = opts.path
    folderName = opts.train_folder
    train_dev_folder = "train_dev_tensor"

    # get the training data from here
    # 0-split_idx becomes the new training data
    folder_path = "%s/%s" %(dataPath,folderName)
    entities  = joblib.load(folder_path+"/train_entities.joblib")[split_idx:]
    relations = joblib.load(folder_path+"/train_relations.joblib")[split_idx:]

    if opts.neg_joint:
        neg_data_tf = joblib.load(folder_path+"/train_neg_samples_joint_tf.joblib")[split_idx:]
        neg_data_mf = joblib.load(folder_path+"/train_neg_samples_joint_mf.joblib")[split_idx:]
        

    neg_data = joblib.load(folder_path+"/train_neg_samples.joblib")[split_idx:]


    # check if the train-dev folder exists inside dataPath. if not, travel to dataPath and mkdir it.
    # also don't forget to chdir back to the original path
    currPath = os.getcwd()
    dev_folder_path = "%s/%s" %(dataPath, train_dev_folder)
    if not os.path.exists(dev_folder_path):
        os.chdir(dataPath)
        os.system("mkdir "+train_dev_folder)
        os.chdir(currPath)


    joblib.dump(entities, dev_folder_path +"/train_entities.joblib")
    joblib.dump(relations,dev_folder_path +"/train_relations.joblib")

    if opts.neg_joint:
        joblib.dump(neg_data_tf, dev_folder_path+"/train_neg_samples_joint_tf.joblib")
        joblib.dump(neg_data_mf, dev_folder_path+"/train_neg_samples_joint_mf.joblib")

    joblib.dump(neg_data, dev_folder_path+"/train_neg_samples.joblib")



if __name__ == '__main__':
    opts = get_params()
    #dump_train_data(opts)
    dump_test_data(opts)
    #create_dev_split_from_train(opts)

