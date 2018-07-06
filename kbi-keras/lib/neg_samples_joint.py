from collections import defaultdict as ddict
from func import *

'''
    batch -> (e1, r, e2) triplets
    neg_samples -> number of negative samples required
    num_entities -> number of entities
    num_entity_pairs -> number of entity_pairs
    entityPairToId -> a dictionary storing the IDs for (e1, e2) [stored as a single number]
'''
def gen_neg_joint(batch, neg_samples, num_entities, num_entity_pairs, entityPairToId):
    batch_len = len(batch)
    es, eo = read_data(batch)
    print("Done reading batch...")
    allEntities = np.arange(0, num_entities)

    def create_random_vector(batch_example):
        e1, r, e2 = batch_example
        batch_example_str= map(str, batch_example)
        s_r = ("\t").join(batch_example_str[:2]) # (e1, r)
        a1 = eo[s_r] # a1 = possible e2s in training seen with (e1, r)
        negative_e2s = np.setdiff1d(allEntities,a1, assume_unique = True) # list of all e2s such that (e1, r, e2) is not a training data point 

        e2_compatible = [e2 for e2 in negative_e2s if (e1, e2) in entityPairToId] # list of all e2s such that (e1, r, e2) is not a training point and (e1, e2) coexist as a pair
        e2_compatible_pairs = [entityPairToId[(e1, e2)] for e2 in e2_compatible] 
        e2_remaining  = [e2 for e2 in negative_e2s if (e1, e2) not in entityPairToId]
        deficit = neg_samples - len(e2_compatible)
        if (deficit <= 0):
            sample_from_compatible =  np.random.choice(e2_compatible, neg_samples, replace=False)
            sampled_pairs = [entityPairToId[(e1, e2)] for e2 in sample_from_compatible]
            return sample_from_compatible, sampled_pairs

        else:
            sample_from_remaining = np.random.choice(e2_remaining, deficit, replace = False)
            sampled_pairs = [entityPairToId[(e1, e2)] for e2 in e2_compatible]
            # append OOVs
            while (len(sampled_pairs) != neg_samples):
                sampled_pairs.append(num_entity_pairs)

            return np.concatenate([e2_compatible, sample_from_remaining]), sampled_pairs

    batchSize = len(batch)
    neg_samples_tf = np.zeros((batchSize, neg_samples))
    neg_samples_mf = np.zeros((batchSize, neg_samples))
    for i, batch_example in enumerate(batch):
        neg_samples_tf[i], neg_samples_mf[i] = create_random_vector(batch_example)
        print "\r>> Processed %d/%d batches"%(i+1, batchSize),
        sys.stdout.flush()
    
    return neg_samples_tf, neg_samples_mf 
