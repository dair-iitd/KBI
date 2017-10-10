import theano
import theano.tensor as T


# == Wrappers around theano functions for manipulating tensors

def get_cross(i, neg_samples):
    def cross_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:,i]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1)*id_ei
    return lambda X: cross_fn(X[0], X[1])


def cross_e1_e2(X):
    e1 = X[:,0]
    e2 = X[:,1]
    return e1*e2

def get_minus(i, neg_samples):
    def minus_fn(entity_vecs, entity_negative_vecs):
        ei = entity_vecs[:,i]
        id_ei = entity_negative_vecs[:, i*neg_samples : (i+1)*neg_samples]
        return ei.dimshuffle(0,'x',1) - id_ei
    return lambda X: minus_fn(X[0], X[1])

def minus_e1_e2(X):
    e1 = X[:,0]
    e2 = X[:,1]
    return e1 - e2

def get_dot(i):
    def dot_fn(relation_vecs, entity_vecs):
        ei = entity_vecs[:,i]
        dotProd = T.batched_dot(relation_vecs, ei)
        return dotProd
    return lambda X: dot_fn(X[0], X[1])

def get_dot_neg(i, neg_samples):
    def dot_fn(relation_vecs, entity_negative_vecs):
        id_ei = entity_negative_vecs[:, i*neg_samples: (i+1)*neg_samples]
        return T.batched_dot(id_ei,relation_vecs)
    return lambda X: dot_fn(X[0], X[1]) 


def inner_prod(x1, x2, x3):
    return T.batched_dot(x1*x2, x3)

def hermitian_product(X):
    e_real, e_im, r_real, r_im = X
    
    e1_real = e_real[:, 0]; e2_real = e_real[:,1]
    e1_im = e_im[:, 0]; e2_im = e_im[:, 1] 
    
    return inner_prod(e1_real, e2_real, r_real) + inner_prod(e1_im, e2_im, r_real) + inner_prod(e1_real, e2_im, r_im) - inner_prod(e1_im, e2_real, r_im)


# == Implementation of Loss functions (softmax and max margin) : Softmax is approximate (uses negative sampling)


def softmax_approx(input):
    score_positive, score_negative = input
    ''' f(e1, r, e2) = r.T(e1*e2) '''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    max_denom = T.max(score_negative, axis = 1, keepdims=True)
    score_negative = T.exp(score_negative - max_denom).sum(axis=1)

    numer = score_positive - max_denom.dimshuffle(0)
    net_score= score_positive - T.log(score_negative)
    return -1*net_score

def max_margin(input):
    score_positive, score_negative = input
    net_score = T.sum(T.maximum(0,1.0 + score_negative.dimshuffle(0,1) - score_positive.dimshuffle(0,'x') ) )
    return net_score



# == Some generic helpers
def lossFn(y_true, y_pred):
    return T.mean(y_pred)


def get_optimizer(opts):
    optimizer = opts.optimizer
    if optimizer=='Adagrad':
        alg = Adagrad(lr=opts.lr)
    elif optimizer=='RMSprop':
        alg = RMSprop(lr=opts.lr)
    elif optimizer=='Adam':
        alg = Adam(lr=opts.lr)
    else:
        print("This optimizer is currently not supported. Modify models.py if you wish to add it")
        sys.exit(1)
    
    return alg

def get_loss_fn(opts):
    if opts.loss == "ll":
        print "Using log likelihood based loss!"
        return softmax_approx
    elif opts.loss == "mm":
        print "using max-margin loss!"
        return max_margin
