def custom_fn(input):
    e1_cross_e2      = input[0]
    e1_cross_e2_prime   = input[1]
    e1_prime_cross_e2   = input[2]
    relation_vecs = input[3]

    ''' f(e1, r, e2) = r.T(e1*e2) '''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    denom_e1 = T.batched_dot(e1_cross_e2_prime, relation_vecs)
    max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
    denom_e1 = T.exp(denom_e1 - max_denom_e1).sum(axis=1)

    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = T.batched_dot(e1_prime_cross_e2, relation_vecs)
    max_denom_e2 = T.max(denom_e2, axis=1, keepdims=True)
    denom_e2 = T.exp(denom_e2 - max_denom_e2).sum(axis=1)

    numer = T.batched_dot(relation_vecs, e1_cross_e2)
    numer = 2*numer - max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
    net_score= numer - T.log(denom_e1) -T.log(denom_e2)
    return -1*net_score

def custom_fn_distMult_max_margin(input):
    e1_cross_e2      = input[0]
    e1_cross_e2_prime   = input[1]
    e1_prime_cross_e2   = input[2]
    relation_vecs = input[3]
    
    denom_e1 = T.batched_dot(e1_cross_e2_prime, relation_vecs)
    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = T.batched_dot(e1_prime_cross_e2, relation_vecs)
    numer = T.batched_dot(relation_vecs, e1_cross_e2)

    net_loss =  T.sum(T.maximum(0,1.0 - numer.dimshuffle(0,'x') +(denom_e1.dimshuffle(0,1))))
    net_loss += T.sum(T.maximum(0,1.0 - numer.dimshuffle(0,'x') +(denom_e2.dimshuffle(0,1)))) #margin = 1.0

    #TransE paper by bordes min (1+numer-denom)
    #Link: https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
    return 1*net_loss



def L2_norm(data,axis):
    return T.sqrt(T.sum(T.sqr(data),axis))

def L1_norm(data,axis):
    return T.sum(data,axis)

def custom_fn_TransE_max_margin(input): 
    e1_minus_e2      = input[0]
    e1_minus_e2_prime   = input[1]
    e1_prime_minus_e2   = input[2]
    relation_vecs = input[3]
    ''' f(e1, r, e2) = r.T(e1*e2) '''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    denom_e1 = L2_norm(e1_minus_e2_prime + relation_vecs.dimshuffle(0,'x',1) , 2)
    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = L2_norm(e1_prime_minus_e2 + relation_vecs.dimshuffle(0,'x',1),2)
    numer = L2_norm(relation_vecs + e1_minus_e2, 1)

    net_loss =  T.sum(T.maximum(0,1.0 + numer.dimshuffle(0,'x') -(denom_e1.dimshuffle(0,1))))
    net_loss += T.sum(T.maximum(0,1.0 + numer.dimshuffle(0,'x') -(denom_e2.dimshuffle(0,1)))) #margin = 1.0

    #TransE paper by bordes min (1+numer-denom)
    #Link: https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
    return 1*net_loss

def custom_fn_TransE(input):
    e1_minus_e2      = input[0]
    e1_minus_e2_prime   = input[1]
    e1_prime_minus_e2   = input[2]
    relation_vecs = input[3]

    ''' f(e1, r, e2) = r.T(e1*e2) '''
    '''denom_e1 = sum{j=1..200}exp(f(e1, r, e2j)) where e2j is a negative sample '''
    numer = L2_norm(e1_minus_e2 + relation_vecs, 1).dimshuffle(0,'x')   
    denom_e1 = numer - L2_norm(e1_minus_e2_prime + relation_vecs.dimshuffle(0,'x',1),2).dimshuffle(0,1)
    max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
    denom_e1 = T.exp(denom_e1 - max_denom_e1).sum(axis=1)

    '''denom_e2 = sum{j=1..200}exp(f(e1j, r, e2)) where e1j is a negative sample '''
    denom_e2 = numer - L2_norm(e1_prime_minus_e2 + relation_vecs.dimshuffle(0,'x',1),2).dimshuffle(0,1)
    max_denom_e2 = T.max(denom_e2, axis = 1, keepdims=True)
    denom_e2 = T.exp(denom_e2 - max_denom_e2).sum(axis=1)

    net_score= max_denom_e1 + max_denom_e2 + T.log(denom_e1) + T.log(denom_e2)
    return net_score


def custom_fn_E(input):
    rs_dot_e1   = input[0]
    ro_dot_e2   = input[1]
    rs_dot_e1_prime= input[2]
    ro_dot_e2_prime= input[3]
    '''denom_e1 = sum{j=1..200}dot(r_s, e1) + dot(r_o, e2j') '''
    denom_e1 = ro_dot_e2_prime + rs_dot_e1.dimshuffle(0,'x')
    max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
    denom_e1 = T.exp(denom_e1 - max_denom_e1).sum(axis=1)

    '''denom_e2 = sum{j=1..200}dot(r_s, e1j') + dot(r_o, e2j) '''
    denom_e2 = rs_dot_e1_prime + ro_dot_e2.dimshuffle(0,'x')
    max_denom_e2 = T.max(denom_e2, axis=1, keepdims=True)
    denom_e2 = T.exp(denom_e2 - max_denom_e2).sum(axis=1)

    numer = rs_dot_e1 + ro_dot_e2
    numer = 2*numer - max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
    net_score= numer - T.log(denom_e1) -T.log(denom_e2)
    return -1*net_score

def custom_fn_E_max_margin(input):
    rs_dot_e1   = input[0]
    ro_dot_e2   = input[1]
    rs_dot_e1_prime= input[2]
    ro_dot_e2_prime= input[3]
    '''denom_e1 = sum{j=1..200}dot(r_s, e1) + dot(r_o, e2j') '''
    denom_e1 = ro_dot_e2_prime + rs_dot_e1.dimshuffle(0,'x')
    
    '''denom_e2 = sum{j=1..200}dot(r_s, e1j') + dot(r_o, e2j) '''
    denom_e2 = rs_dot_e1_prime + ro_dot_e2.dimshuffle(0,'x')
    
    numer = rs_dot_e1 + ro_dot_e2
    
    net_score  = T.sum(T.maximum(0,1.0 + denom_e1.dimshuffle(0,1) - numer.dimshuffle(0,'x') ) )
    net_score += T.sum(T.maximum(0,1.0 + denom_e2.dimshuffle(0,1) - numer.dimshuffle(0,'x') ) )
     
    return 1*net_score

def create_E_plus_distMultfn(alpha):
    def custom_fn(input):
        e1_cross_e2    = input[0]
        e1_cross_e2_prime = input[1]
        e1_prime_cross_e2 = input[2]
        relation_vecs  = input[3]
        rs_dot_e1 = input[4]
        ro_dot_e2 = input[5]
        rs_dot_e1_prime = input[6]
        ro_dot_e2_prime = input[7]

        denom_e1 = alpha*T.batched_dot(e1_cross_e2_prime, relation_vecs) + (1.0-alpha)*(ro_dot_e2_prime + rs_dot_e1.dimshuffle(0,'x'))
        max_denom_e1 = T.max(denom_e1, axis = 1, keepdims=True)
        denom_e1 = T.exp(denom_e1 - max_denom_e1).sum(axis=1)

        denom_e2 = alpha*T.batched_dot(e1_prime_cross_e2, relation_vecs) + (1.0-alpha)*(rs_dot_e1_prime + ro_dot_e2.dimshuffle(0,'x'))
        max_denom_e2 = T.max(denom_e2, axis=1, keepdims=True)
        denom_e2 = T.exp(denom_e2 - max_denom_e2).sum(axis=1)

        numer = alpha*T.batched_dot(relation_vecs, e1_cross_e2) + (1.0-alpha)*(rs_dot_e1 + ro_dot_e2)
        numer = 2*numer - max_denom_e1.dimshuffle(0) - max_denom_e2.dimshuffle(0)
        net_score= numer - T.log(denom_e1) -T.log(denom_e2)
        return -1*net_score

    return custom_fn



# def build_TransE(opts):
#     neg_samples     = opts.neg_samples
#     vect_dim        = opts.vect_dim
#     num_entities    = opts.num_entities
#     num_relations   = opts.num_relations
#     warm_start      = opts.warm_start
#     optimizer       = opts.optimizer
    
#     l2_reg_entity   = opts.l2_entity
#     l2_reg_relation = opts.l2_relation
#     #define all inputs
#     kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
#     kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
#     neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')

#     e1_minus_e2, e1_minus_e2_prime, e1_prime_minus_e2, relation_vectors  = getTransE_score(kb_entities, kb_relations, neg_samples_kb, opts)
    
#     #score_kb  = merge([e1_minus_e2, e1_minus_e2_prime, e1_prime_minus_e2, relation_vectors], mode = custom_fn_TransE_max_margin, output_shape = (None,1))
#     score_kb  = merge([e1_minus_e2, e1_minus_e2_prime, e1_prime_minus_e2, relation_vectors], mode = custom_fn_TransE, output_shape = (None,1)) 
#     alg = get_optimizer(opts)

#     model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
#     model.compile(loss = lossFn, optimizer=alg)

#     aux_model = None

#     return aux_model, model


# def build_E_plus_distMult(opts):
#     neg_samples   = opts.neg_samples
#     vect_dim      = opts.vect_dim
#     num_entities  = opts.num_entities
#     num_relations = opts.num_relations
#     warm_start    = opts.warm_start
#     optimizer     = opts.optimizer
#     l2_reg = opts.l2    
#     alpha  = opts.alpha
#     #define all inputs
#     kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
#     kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
#     neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')

#     entities    = Embedding(output_dim=vect_dim, input_dim=num_entities, init='normal',name = 'entity_embeddings', W_regularizer=l2(l2_reg))
#     entities_E  = Embedding(output_dim=vect_dim, input_dim=num_entities, init='normal',name = 'entity_embeddings_E', W_regularizer=l2(l2_reg))
#     relations   = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings', W_regularizer=l2(l2_reg))
#     relations_s = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_s', W_regularizer=l2(l2_reg))
#     relations_o = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_o', W_regularizer=l2(l2_reg))


#     entity_vectors = entities(kb_entities)
#     entity_negative_vectors = entities(neg_samples_kb)
#     relation_vectors = Flatten()(relations(kb_relations))

#     entity_vectors_E = entities_E(kb_entities)
#     entity_negative_vectors_E = entities_E(neg_samples_kb)
#     relations_s_vectors = Flatten()(relations_s(kb_relations))
#     relations_o_vectors = Flatten()(relations_o(kb_relations))

#     get_cross_1 = get_cross(0, neg_samples)
#     get_cross_2 = get_cross(1, neg_samples)
#     get_dot_1 =  get_dot(0)
#     get_dot_2 =  get_dot(1)
#     get_dot_neg_1 = get_dot_neg(0, neg_samples)
#     get_dot_neg_2 = get_dot_neg(1, neg_samples)

#     #E model 
#     rs_dot_e1 = merge([relations_s_vectors, entity_vectors_E], mode =get_dot_1, output_shape = (None,1))
#     ro_dot_e2 = merge([relations_o_vectors, entity_vectors_E], mode =get_dot_2, output_shape = (None,1))
#     rs_dot_e1_prime = merge([relations_s_vectors, entity_negative_vectors_E], mode =get_dot_neg_2, output_shape = (None, neg_samples))
#     ro_dot_e2_prime = merge([relations_o_vectors, entity_negative_vectors_E], mode =get_dot_neg_1, output_shape = (None, neg_samples))

#     #DistMult model
#     e1_cross_e2_prime = merge([entity_vectors, entity_negative_vectors], mode = get_cross_1, output_shape = (None, neg_samples, vect_dim))
#     e1_prime_cross_e2 = merge([entity_vectors, entity_negative_vectors], mode = get_cross_2, output_shape = (None, neg_samples, vect_dim))
#     e1_cross_e2    = Lambda(cross_e1_e2, output_shape = (vect_dim,))(entity_vectors)

#     custom_fn_E_plus_distMult = create_E_plus_distMultfn(alpha)
#     score_kb  = merge([e1_cross_e2, e1_cross_e2_prime, e1_prime_cross_e2, relation_vectors, rs_dot_e1, ro_dot_e2, rs_dot_e1_prime, ro_dot_e2_prime], mode = custom_fn_E_plus_distMult, output_shape = (None,1))
#     alg = get_optimizer(opts)

#     model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
#     model.compile(loss = lossFn, optimizer=alg)

#     if opts.typing:
#         aux_model = build_typeNet(opts, entities)
#     else:
#         aux_model = None
#     return aux_model, model





# def build_EModel(opts):
#     neg_samples   = opts.neg_samples
#     vect_dim      = opts.vect_dim
#     num_entities  = opts.num_entities
#     num_relations = opts.num_relations
#     warm_start    = opts.warm_start
#     optimizer     = opts.optimizer
#     l2_reg = opts.l2_entity    
    
#     #define all inputs
#     kb_entities     = Input(shape=(2,), dtype='int32', name='kb_entities')
#     kb_relations    = Input(shape=(1,), dtype='int32', name='kb_relations')
#     neg_samples_kb  = Input(shape=(2*neg_samples, ), dtype = 'int32', name='kb_neg_examples')

#     #define embeddings
#     entities    = Embedding(output_dim=vect_dim, input_dim=num_entities, init='normal',name = 'entity_embeddings',  W_regularizer=l2(l2_reg))
#     relations_s = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_s', W_regularizer=l2(l2_reg))
#     relations_o = Embedding(output_dim=vect_dim, input_dim=num_relations, input_length=1,init='normal', name='relation_embeddings_o', W_regularizer=l2(l2_reg))

#     entity_vectors = entities(kb_entities)
#     entity_negative_vectors = entities(neg_samples_kb)
#     relation_vectors_s = Flatten()(relations_s(kb_relations))
#     relation_vectors_o = Flatten()(relations_o(kb_relations))

#     get_dot_1 =  get_dot(0)
#     get_dot_2 =  get_dot(1)
#     get_dot_neg_1 = get_dot_neg(0, neg_samples)
#     get_dot_neg_2 = get_dot_neg(1, neg_samples)
#     rs_dot_e1 = merge([relation_vectors_s, entity_vectors], mode =get_dot_1, output_shape = (None,))
#     ro_dot_e2 = merge([relation_vectors_o, entity_vectors], mode =get_dot_2, output_shape = (None,))
#     rs_dot_e1_prime = merge([relation_vectors_s, entity_negative_vectors], mode =get_dot_neg_2, output_shape = (None, neg_samples))
#     ro_dot_e2_prime = merge([relation_vectors_o, entity_negative_vectors], mode =get_dot_neg_1, output_shape = (None, neg_samples))

#     if opts.loss == "ll":
#         score_kb  = merge([rs_dot_e1, ro_dot_e2, rs_dot_e1_prime, ro_dot_e2_prime], mode = custom_fn_E, output_shape = (None,1))
#     elif opts.loss == "mm":
#         print "Using Max-margin loss!"
#         score_kb  = merge([rs_dot_e1, ro_dot_e2, rs_dot_e1_prime, ro_dot_e2_prime], mode = custom_fn_E_max_margin, output_shape = (None,1))
        
#     alg = get_optimizer(opts)

#     model = Model(input=[kb_entities, kb_relations, neg_samples_kb], output=score_kb)
#     model.compile(loss = lossFn, optimizer=alg)
#     if opts.typing:
#         aux_model = build_typeNet(opts, entities)
#     else:
#         aux_model = None

#     return aux_model, model






