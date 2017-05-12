def scoreDM_OOV(model, opts, testData, trainData, oov_flags_1, oov_flags_2, seen_e2, oov_e2, distMult_oovEval, distMult_get_scores, distMult_run, oov_flags = None):
    print("evaluating.")
    train_entities, train_relations = trainData   
    for layer in model.layers:
        if layer.name == 'entity_embeddings' or layer.name == 'entity_embeddings_DM':
            entity_weights = layer.get_weights()[0]
        elif layer.name == 'relation_embeddings' or layer.name == 'relation_embeddings_DM':
            relation_weights = layer.get_weights()[0]
    
    vect_dim = entity_weights.shape[1]
    if opts.oov_train:
        oov_embedding = entity_weights[-1] 
    elif opts.oov_average:
            oov_embedding = create_oov_vector(entity_weights, train_entities)
    else:
        print "using random oov vector"
        oov_embedding = np.random.randn(vect_dim)
   
    if os.path.exists("ranks_FB15K_DM.joblib"):
        ranks = joblib.load("ranks_FB15K_DM.joblib")

    else:
        ranks = distMult_oovEval(entity_weights, relation_weights, testData, seen_e2, oov_flags_1, oov_flags_2, oov_embedding)

        joblib.dump(ranks, "ranks_FB15K_DM.joblib")

    for i, oov_comparison_bit in enumerate(ranks[1]):
        if (not oov_comparison_bit and not oov_flags_2[i]):
            ranks[0][i] += oov_e2[i]

    def get_best(score_vector, seen_e2_curr, e2):
        best_pred = None
        best_score = -1<<20
    
        for i, score in enumerate(score_vector):
            if i not in seen_e2_curr or i == e2:
                if (score > best_score):
                    best_score = score; best_pred = i 


        all_best = []

        for i, score in enumerate(score_vector):
            if score == best_score: all_best.append(i)

        return all_best, best_score

    def get_preds(rank_id_pairs):
        test_curr = []; seen_e2_curr = []; oov_flags_1_curr = []

        for (rank, i) in rank_id_pairs:
            test_curr.append(testData[i]); 
            seen_e2_curr.append(seen_e2[i]); 
            oov_flags_1_curr.append(oov_flags_1[i]); 

        score_matrix = distMult_get_scores(entity_weights, relation_weights, test_curr, oov_flags_1_curr, oov_embedding)
        preds = []

        for i, score_vector in enumerate(score_matrix):
            e1, r, e2 = test_curr[i]
            best_pred = get_best(score_vector, seen_e2_curr[i], e2)
            preds.append(best_pred)

        return preds

    def get_inverse_scores(inv_relation_arr):
        test_curr = []
        for i, test_point in enumerate(testData):
            e1, r, e2 = test_point
            inv_score, inv_r = inv_relation_arr[i]
            if (inv_r is not  None):
                test_curr.append(np.array([e2, inv_r, e1]))
            else:
                test_curr.append(np.array([e1, r, e2]))


        score_distMult = distMult_run(entity_weights, relation_weights, test_curr)
        return score_distMult
        

    symmetric_r, inv_relation_arr = get_inverse_examples(train_entities, train_relations, testData)

    entityIdToName = get_id_to_name(1, 2, "lib/ent_wiki_mapping.txt")
    relationIdToName = get_id_to_name(0, 1, "lib/relation_id.txt")

    mrr  = np.mean(1.0/ranks[0])
    hits = np.mean(ranks[0] <= 10.0)    

    print("mrr: %5.4f, hits@10: %5.4f" %(mrr, hits))

    entity_freq = ddict(int)
    e1_mrr_dict = ddict(list)

    for i, (e1,e2) in enumerate(train_entities): 
        entity_freq[e1] += 1

    for i, test_point in enumerate(testData):
        e1, r, e2 = test_point
        e1_mrr_dict[e1].append(1.0/ranks[0][i])

   

    f = open("mrr_freq.txt", "w")
    for ent in e1_mrr_dict:

        avg_mrr = sum(e1_mrr_dict[ent])/float(len(e1_mrr_dict[ent]))
        f.write("%s: %d, %5.4f\n" %(entityIdToName[ent], entity_freq[ent], avg_mrr))
           
    f.close() 

    f = open("inverse_relations.txt", "w")
    for i, test_point in enumerate(testData):
       e1, r1, e2 = test_point
       score, r2 = inv_relation_arr[i]
       if r2 is not None:
        cosine_dist = sp.spatial.distance.cosine(relation_weights[r1], relation_weights[r2])
        f.write("%s \t %s : %d: %5.4f\n" %(relationIdToName[r1], relationIdToName[r2], score, cosine_dist))
        
    f.close() 


    cosine_dist = sp.spatial.distance.cosine(entity_weights[6731], entity_weights[3474])
    print(cosine_dist)


    return np.mean(1.0/ranks[0]), np.mean(ranks[0] <= 10.0)    