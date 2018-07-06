import argparse
import sys

def get_params():
    parser = argparse.ArgumentParser(description = 'Atomic model')    
    parser.add_argument('-neg_samples', action="store", default=200, dest="neg_samples", type=int)
    parser.add_argument('-vect_dim', action="store", default=100, dest="vect_dim", type=int)
    parser.add_argument('-num_entities', action="store", default=14541, dest="num_entities", type=int)
    parser.add_argument('-num_entity_pairs', action="store", default=14541, dest="num_entity_pairs", type=int)
    parser.add_argument('-batch_size', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-num_relations', action="store", default=237, dest="num_relations", type=int)
    parser.add_argument('-epochs', action="store", default=50, dest="nb_epochs", type=int)
    parser.add_argument('-warm_start', action="store", default=False, dest="warm_start", type=bool)
    parser.add_argument('-model', action="store", default="deepDistMult", dest="model", type=str)
    parser.add_argument('-dataset', action="store", default="nytAtomic", dest="dataset", type=str)
    parser.add_argument('-optimizer', action="store", default="Adagrad", dest="optimizer", type=str)
    parser.add_argument('-rate', action="store", default=0.5, dest="lr", type=float)
    parser.add_argument('-l2_entity', action="store", default=0, dest="l2_entity", type=float)
    parser.add_argument('-l2_entity_E', action="store", default=0, dest="l2_entity_E", type=float)
    parser.add_argument('-l2_entity_pair', action="store", default=0, dest="l2_entity_pair", type=float)

    parser.add_argument('-l2_relation', action="store", default=0, dest="l2_relation", type=float)
    parser.add_argument('-l2_relation_MF', action="store", default=0, dest="l2_relation_MF", type=float)


    parser.add_argument('-unit_norm', action="store", default=1, dest="unit_norm_reg", type=int)
    parser.add_argument('-train', action="store", default=1, dest="train", type=int)
    parser.add_argument('-typing', action="store", default=0, dest="typing", type=int)
    parser.add_argument('-num_types', action="store", default=4054, dest="num_types", type=int)
    parser.add_argument('-type_neg_samples', action="store", default=50, dest="type_neg_samples", type=int)
    parser.add_argument('-type_weight', action="store", default=0.25, dest="type_weight", type=float)
    parser.add_argument('-oov_eval', action="store", default=1, dest="oov_eval", type=int)
    parser.add_argument('-oov_train', action="store", default=0, dest="oov_train", type=int)
    parser.add_argument('-oov_avg', action="store", default=0, dest="oov_average", type=int)
    parser.add_argument('-eval_after', action="store", default=50, dest="eval_after", type=int)
    parser.add_argument('-eval_every', action="store", default=10, dest="eval_every", type=int)
    parser.add_argument('-shared_r', action="store", default=1, dest="shared_r", type=int)
    parser.add_argument('-add_tanh', action="store", default=0, dest="add_tanh", type=int)
    parser.add_argument('-norm_score', action="store", default=0, dest="normalize_score", type=int)
    parser.add_argument('-norm_score_eval', action="store", default=0, dest="normalize_score_eval", type=int)

    parser.add_argument('-static_alpha', action="store", default=0, dest="static_alpha", type=int)
    parser.add_argument('-alphaMF', action="store", default=1.0, dest="alphaMF", type=float)
    parser.add_argument('-static_beta', action="store", default=0, dest="static_beta", type=int)
    parser.add_argument('-add_loss', action="store", default=0, dest="add_loss", type=int)


    parser.add_argument('-evalDev', action="store", default=1, dest="evalDev", type=int)

    parser.add_argument('-dropout_DM', action="store", default=0.0, dest="dropout_DM", type=float)
    parser.add_argument('-dropout_MF', action="store", default=0.0, dest="dropout_MF", type=float)

    parser.add_argument('-model_path', action="store", default="", dest="model_path", type=str)
    parser.add_argument('-loss', action="store", default="ll", dest="loss", type=str)

    opts = parser.parse_args(sys.argv[1:])
    print "negative samples", opts.neg_samples
    print "training?", opts.train
    print "oov evaluation", opts.oov_eval
    print "typing?", opts.typing
    print "unit entity norm?", opts.unit_norm_reg
    print "num_types", opts.num_types
    print "type_neg_samples", opts.type_neg_samples
    print "batch size", opts.batch_size
    print "embedding dimension", opts.vect_dim
    print "num_entities", opts.num_entities
    if opts.model == "MF":
        print "num entity pairs", opts.num_entity_pairs
        
    print "num_relations", opts.num_relations
    print "num_epochs", opts.nb_epochs
    print "warm_start", opts.warm_start
    print "model", opts.model
    print "dataset folder", opts.dataset
    print "optimizer", opts.optimizer
    print "learning rate", opts.lr
    print "l2 regularization entities", opts.l2_entity
    print "l2 regularization entity pairs", opts.l2_entity_pair
    print "l2 regularization relations", opts.l2_relation
    print "l2 regularization relations MF", opts.l2_relation_MF
    

    return opts

