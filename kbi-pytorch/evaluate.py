import utils
import numpy
import torch
import time
import gc


class ranker(object):
    """
    A network that ranks entities based on a scoring function. It excludes entities that have already
    been seen in any kb to compute the ranking as in ####### cite paper here ########. It can be constructed
    from any scoring function/model from models.py
    """
    def __init__(self, scoring_function, all_kb):
        """
        The initializer\n
        :param scoring_function: The model function to used to rank the entities
        :param all_kb: A union of all the knowledge bases (train/test/valid)
        """
        super(ranker, self).__init__()
        self.scoring_function = scoring_function
        self.all_kb = all_kb
        self.knowns_o = {} #o seen w/t s,r
        self.knowns_s = {} 
        print("building all known database from joint kb")
        for fact in self.all_kb.facts:
            if (fact[0], fact[1]) not in self.knowns_o:
                self.knowns_o[(fact[0], fact[1])] = set()
            self.knowns_o[(fact[0], fact[1])].add(fact[2])

            if (fact[2], fact[1]) not in self.knowns_s:
                self.knowns_s[(fact[2], fact[1])] = set()
            self.knowns_s[(fact[2], fact[1])].add(fact[0])

        print("converting to lists")
        for k in self.knowns_o:
            self.knowns_o[k] = list(self.knowns_o[k])
        for k in self.knowns_s:
            self.knowns_s[k] = list(self.knowns_s[k])
        print("done")

    def get_knowns(self, e, r, flag_s_o=0):
        """
        computes and returns the set of all entites that have been seen as a fact (s, r, _) or (_, r, o)\n
        :param e: The head(s)/tail(o) of the fact
        :param r: The relation of the fact
        :param flag_s_o: whether e is s #s is fixed we try o
        :return: All entites o such that (s, r, o) has been seen in all_kb 
                 OR
                 All entites s such that (s, r, o) has been seen in all_kb 
        """
        if flag_s_o:
            ks = [self.knowns_o[(a, b)] for a,b in zip(e, r)]
        else:
            ks = [self.knowns_s[(a, b)] for a,b in zip(e, r)]
        lens = [len(x) for x in ks]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens-len(x)), 'edge') for x in ks]
        result= numpy.array(ks)
        return result

    def forward(self, s, r, o, knowns, flag_s_o=0):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :return: rank of o, score of each entity and score of the gold o
        """
        if flag_s_o:
            scores = self.scoring_function(s, r, None).data
            score_of_expected = scores.gather(1, o.data)
        else:
            scores = self.scoring_function(None, r, o).data
            score_of_expected = scores.gather(1, s.data)

        scores.scatter_(1, knowns, self.scoring_function.minimum_value)
        greater = scores.ge(score_of_expected).float()
        equal = scores.eq(score_of_expected).float()
        rank = greater.sum(dim=1)+1+equal.sum(dim=1)/2.0
        return rank, scores, score_of_expected


def evaluate(name, ranker, kb, batch_size, verbose=0, top_count=5, hooks=None):
    """
    Evaluates an entity ranker on a knowledge base, by computing mean reverse rank, mean rank, hits 10 etc\n
    Can also print type prediction score with higher verbosity.\n
    :param name: A name that is displayed with this evaluation on the terminal
    :param ranker: The ranker that is used to rank the entites
    :param kb: The knowledge base to evaluate on. Must be augmented with type information when used with higher verbosity
    :param batch_size: The batch size of each minibatch
    :param verbose: The verbosity level. More info is displayed with higher verbosity
    :param top_count: The number of entities whose details are stored
    :param hooks: The additional hooks that need to be run with each mini-batch
    :return: A dict with the mrr, mr, hits10 and hits1 of the ranker on kb
    """
    if hooks is None:
        hooks = []
    totals = {"e2":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "e1":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "m":{"mrr":0, "mr":0, "hits10":0, "hits1":0}}
    start_time = time.time()
    facts = kb.facts
    '''
    if(verbose>0):
        totals["correct_type"]={"e1":0, "e2":0}
        entity_type_matrix = kb.entity_type_matrix.cuda()
        for hook in hooks:
            hook.begin()
    '''
    for i in range(0, int(facts.shape[0]), batch_size):
        start = i
        end = min(i+batch_size, facts.shape[0])
        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]
        knowns_o = ranker.get_knowns(s, r, flag_s_o=1)
        knowns_s = ranker.get_knowns(o, r, flag_s_o=0)
        s = torch.autograd.Variable(torch.from_numpy(s).cuda().unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).cuda().unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).cuda().unsqueeze(1), requires_grad=False)
        knowns_s = torch.from_numpy(knowns_s).cuda()
        knowns_o = torch.from_numpy(knowns_o).cuda()
        
        ranks_o, scores_o, score_of_expected_o = ranker.forward(s, r, o, knowns_o, flag_s_o=1)
        ranks_s, scores_s, score_of_expected_s = ranker.forward(s, r, o, knowns_s, flag_s_o=0)

        #e1,r,?
        totals['e2']['mr'] += ranks_o.sum()
        totals['e2']['mrr'] += (1.0/ranks_o).sum()
        totals['e2']['hits10'] += ranks_o.le(11).float().sum()
        totals['e2']['hits1'] += ranks_o.eq(1).float().sum()
        #?,r,e2 
        totals['e1']['mr'] += ranks_s.sum()
        totals['e1']['mrr'] += (1.0/ranks_s).sum()
        totals['e1']['hits10'] += ranks_s.le(11).float().sum()
        totals['e1']['hits1'] += ranks_s.eq(1).float().sum()

        totals['m']['mr'] += (ranks_s.sum()+ranks_o.sum())/2.0
        totals['m']['mrr'] += ((1.0/ranks_s).sum()+(1.0/ranks_o).sum())/2.0
        totals['m']['hits10'] += (ranks_s.le(11).float().sum() + ranks_o.le(11).float().sum())/2.0
        totals['m']['hits1'] += (ranks_s.eq(1).float().sum() + ranks_o.eq(1).float().sum())/2.0


        extra = ""
        '''
        if verbose > 0:
            scores_s.scatter_(1, s.data, score_of_expected_s)
            top_scores_s, top_predictions_s = scores_s.topk(top_count, dim=-1)
            top_predictions_type_s = torch.nn.functional.embedding(top_predictions_s, entity_type_matrix).squeeze(-1)
            expected_type_s = torch.nn.functional.embedding(s, entity_type_matrix).squeeze()
            correct_s = expected_type_s.eq(top_predictions_type_s[:, 0]).float()
            correct_count_s = correct_s.sum()
            totals["correct_type"]["e1"] += correct_count_s.item()#data[0]
            extra += "TP-s error %5.3f |" % (100*(1.0-totals["correct_type"]["e1"]/end)) #?,r,o
            for hook in hooks:
                hook(s.data, r.data, o.data, ranks_s, top_scores_s, top_predictions_s, expected_type_s, top_predictions_type_s)
        
            scores_o.scatter_(1, o.data, score_of_expected_o)
            top_scores_o, top_predictions_o = scores_o.topk(top_count, dim=-1)
            top_predictions_type_o = torch.nn.functional.embedding(top_predictions_o, entity_type_matrix).squeeze(-1)
            expected_type_o = torch.nn.functional.embedding(o, entity_type_matrix).squeeze()
            correct_o = expected_type_o.eq(top_predictions_type_o[:, 0]).float()
            correct_count_o = correct_o.sum()
            totals["correct_type"]["e2"] += correct_count_o.item()#data[0]
            extra += "TP-o error %5.3f |" % (100*(1.0-totals["correct_type"]["e2"]/end)) #s,r,?
            for hook in hooks:
                hook(s.data, r.data, o.data, ranks_o, top_scores_o, top_predictions_o, expected_type_o, top_predictions_type_o)
        '''
        utils.print_progress_bar(end, facts.shape[0], "Eval on %s" % name, (("|M| mrr:%3.2f|h10:%3.2f%"
                                                                                  "%|h1:%3.2f|e1| mrr:%3.2f|h10:%3.2f%"
                                                                                  "%|h1:%3.2f|e2| mrr:%3.2f|h10:%3.2f%"
                                                                                  "%|h1:%3.2f|time %5.0f|") %
                                 (100.0*totals['m']['mrr']/end, 100.0*totals['m']['hits10']/end,
                                  100.0*totals['m']['hits1']/end, 100.0*totals['e1']['mrr']/end, 100.0*totals['e1']['hits10']/end,
                                  100.0*totals['e1']['hits1']/end, 100.0*totals['e2']['mrr']/end, 100.0*totals['e2']['hits10']/end,
                                  100.0*totals['e2']['hits1']/end, time.time()-start_time)) + extra, color="green")
    
    gc.collect()
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.end()
    print(" ")
            
    totals['m'] = {x:totals['m'][x]/facts.shape[0] for x in totals['m']}
    totals['e1'] = {x:totals['e1'][x]/facts.shape[0] for x in totals['e1']}
    totals['e2'] = {x:totals['e2'][x]/facts.shape[0] for x in totals['e2']}

    return totals
