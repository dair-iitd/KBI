import numpy as np
import time
import theano
import sys, random

def read_data(batch):
    es={};#?ro
    eo={}#sr?
    datSize = len(batch)
    for i,ele in enumerate(batch):
        print "\r>> Processed %d/%d batches"%(i+1, datSize),
        sys.stdout.flush()
        ele = map(str, list(ele))
        
        o=("\t").join(ele[:2])
        s=("\t").join(ele[1:])
        if o in eo:
            eo[o].append(int(ele[-1]))
        else:
            eo[o]=[int(ele[-1])]
        
        if s in es:
            es[s].append(int(ele[0]))
        else:
            es[s]=[int(ele[0])]

    return es, eo

def read_data_pair(batch):
    re={};
    datSize = len(batch)
    for i,ele in enumerate(batch):
        print "\r>> Processed %d/%d batches"%(i+1, datSize),
        sys.stdout.flush()
        
        if ele[0] in re:
            re[ele[0]].add(int(ele[1]))
        else:
            re[ele[0]] = set([int(ele[1])])
        
    return re

def gen_neg_samples(batch, neg_samples, size_entities):
    batch_len = len(batch)
    allEntities = np.arange(0,size_entities)
    es, eo = read_data(batch)
    print("Done reading batch...")
    def create_random_vector(batch_example):
    	batch_example_str= map(str, batch_example)
    	s_r = ("\t").join(batch_example_str[:2])
    	r_o = ("\t").join(batch_example_str[1:])
    	a1 = eo[s_r]; b1 = es[r_o]
    	a = np.setdiff1d(allEntities,a1); b = np.setdiff1d(allEntities,b1)
        return np.concatenate([np.random.choice(a, neg_samples, replace = False),np.random.choice(b, neg_samples, replace = False)])

    batchSize = len(batch)
    negSampleArray = np.zeros((batchSize, 2*neg_samples))
    for i, batch_example in enumerate(batch):
    	negSampleArray[i] = create_random_vector(batch_example)
    	print "\r>> Processed %d/%d batches"%(i+1, batchSize),
    	sys.stdout.flush()
		
    return negSampleArray 

def gen_neg_pair_samples_random(batch, neg_samples, size_entities):
    '''
    Generates negative samples randomly. Do not use train/test/valid data to reject incorrectly picked negative samples
    Suited for very large training data
    Approximate neg data generation
    '''
    batch_len = len(batch)
    allEntities = np.arange(0,size_entities)
    batchSize = len(batch)
    negSampleArray = np.zeros((batchSize, 2*neg_samples))
    for i, batch_example in enumerate(batch):
        negSampleArray[i] = random.sample(allEntities, 2*neg_samples)
        print "\r>> Processed %d/%d batches"%(i+1, batchSize),
        sys.stdout.flush()
        
    return negSampleArray 



def gen_neg_pair_samples_randomUpdate(batch, neg_samples, size_entities, all_data):
    '''
    Suited for very moderately - large training data
    Exact neg data generation
    '''
    batch_len = len(batch)
    allEntities = np.arange(0,size_entities)
    re = read_data_pair(all_data)
    del all_data
    print("Done reading batch...")
    def create_random_vector(batch_example):
        negSample = np.array([])
        r = re[batch_example[0]]
        negSample_counter = 2*neg_samples
        while neg_samples.shape[0] != 2*neg_samples:
            negSample_gen = random.sample(allEntities, negSample_counter)
            neg_samples = np.concatenate([neg_samples, np.array(list(set(negSample_gen).difference(r)))])
            negSample_counter = (2*neg_samples) - neg_samples.shape[0]

        return np.array(neg_samples, dtype='int32')   
        
    batchSize = len(batch)
    negSampleArray = np.zeros((batchSize, 2*neg_samples))
    for i, batch_example in enumerate(batch):
        negSampleArray[i] = create_random_vector(batch_example)
        print "\r>> Processed %d/%d batches"%(i+1, batchSize),
        sys.stdout.flush()
        
    return negSampleArray 

def gen_neg_pair_samples(batch, neg_samples, size_entities, all_data):
    '''
    Suited for small train data
    Exact neg data generation
    '''
    batch_len = len(batch)
    allEntities = np.arange(0,size_entities)
    re = read_data_pair(all_data)
    del all_data
    print("Done reading batch...")
    def create_random_vector(batch_example):
        a1 = re[batch_example[0]]
        a = np.setdiff1d(allEntities,a1)
        return np.random.choice(a, 2*neg_samples, replace = False)

    batchSize = len(batch)
    negSampleArray = np.zeros((batchSize, 2*neg_samples))
    for i, batch_example in enumerate(batch):
        negSampleArray[i] = create_random_vector(batch_example)
        print "\r>> Processed %d/%d batches"%(i+1, batchSize),
        sys.stdout.flush()
        
    return negSampleArray 
