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
        self.knowns = {}
        print("building all known database from joint kb")
        for fact in self.all_kb.facts:
            if (fact[0], fact[1]) not in self.knowns:
                self.knowns[(fact[0], fact[1])] = set()
            self.knowns[(fact[0], fact[1])].add(fact[2])
        print("converting to lists")
        for k in self.knowns:
            self.knowns[k] = list(self.knowns[k])
        print("done")

    def get_knowns(self, s, r):
        """
        computes and returns the set of all entites that have been seen as a fact (s, r, _)\n
        :param s: The head of the fact
        :param r: The relation of the fact
        :return: All entites o such that (s, r, o) has been seen in all_kb
        """
        ks = [self.knowns[(a, b)] for a,b in zip(s, r)]
        lens = [len(x) for x in ks]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens-len(x)), 'edge') for x in ks]
        resutl= numpy.array(ks)
        return resutl

    def forward(self, s, r, o, knowns):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :return: rank of o, score of each entity and score of the gold o
        """
        scores = self.scoring_function(s, r, None).data
        score_of_expected = scores.gather(1, o.data)
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
    totals = {"mrr":0, "mr":0, "hits10":0, "hits1":0}
    start_time = time.time()
    facts = kb.facts
    if(verbose>0):
        totals["correct_type"] = 0
        entity_type_matrix = kb.entity_type_matrix.cuda()
        for hook in hooks:
            hook.begin()
    for i in range(0, int(facts.shape[0]), batch_size):
        start = i
        end = min(i+batch_size, facts.shape[0])
        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]
        knowns = ranker.get_knowns(s, r)
        s = torch.autograd.Variable(torch.from_numpy(s).cuda().unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).cuda().unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).cuda().unsqueeze(1), requires_grad=False)
        knowns = torch.from_numpy(knowns).cuda()
        
        ranks, scores, score_of_expected = ranker.forward(s, r, o, knowns)
        totals['mr'] += ranks.sum()
        totals['mrr'] += (1.0/ranks).sum()
        totals['hits10'] += ranks.le(11).float().sum()
        totals['hits1'] += ranks.eq(1).float().sum()
        
        extra = ""
        if verbose > 0:
            scores.scatter_(1, o.data, score_of_expected)
            top_scores, top_predictions = scores.topk(top_count, dim=-1)
            top_predictions_type = torch.nn.functional.embedding(top_predictions, entity_type_matrix).squeeze(-1)
            expected_type = torch.nn.functional.embedding(o, entity_type_matrix).squeeze()
            correct = expected_type.eq(top_predictions_type[:, 0]).float()
            correct_count = correct.sum()
            totals["correct_type"] += correct_count[0]
            extra += " TP error %5.3f |" % (100*(1.0-totals["correct_type"]/end))
            for hook in hooks:
                hook(s.data, r.data, o.data, ranks, top_scores, top_predictions, expected_type, top_predictions_type)
        
        utils.print_progress_bar(end, facts.shape[0], "Evaluating on %s" % name, (("| mrr:%8.5f | mr:%10.5f | h10:%7.3f%"
                                                                                  "% | h1:%7.3f%%| time %5.0f |") %
                                 (totals['mrr']/end, totals['mr']/end, 100.0*totals['hits10']/end,
                                  100.0*totals['hits1']/end, time.time()-start_time)) + extra, color="green")
    gc.collect()
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.end()
    return {x:totals[x]/facts.shape[0] for x in totals}
