import numpy
import time
import evaluate
import torch
import kb
import utils
import os


class Trainer(object):
    def __init__(self, scoring_function, regularizer, loss, optim, train, valid, test, verbose=0, batch_size=1000,
                 hooks=None , eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs"):
        super(Trainer, self).__init__()
        self.scoring_function = scoring_function
        self.loss = loss
        self.regularizer = regularizer
        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid":{"mrr":0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []

    def step(self):
        s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)
        fp = self.scoring_function(s, r, o)
        fns = self.scoring_function(ns, r, o)
        fno = self.scoring_function(s, r, no)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0
        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg
        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches, valid_score, test_score):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches*self.batch_size/self.train.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['valid_score_e2'] = valid_score['e2']
        state['test_score_e2'] = test_score['e2']
        state['valid_score_e1'] = valid_score['e1']
        state['test_score_e1'] = test_score['e1']
        state['valid_score_m'] = valid_score['m']
        state['test_score_m'] = test_score['m']
        filename = os.path.join(self.save_directory, "epoch_%.1f_val_%5.2f_%5.2f_%5.2f_test_%5.2f_%5.2f_%5.2f.pt"%(state['epoch'],
                                                                                           state['valid_score_e2']['mrr'], 
                                                                                           state['valid_score_e1']['mrr'], 
                                                                                           state['valid_score_m']['mrr'],
                                                                                           state['test_score_e2']['mrr'],
                                                                                           state['test_score_e1']['mrr'],
                                                                                           state['test_score_m']['mrr']))


        #torch.save(state, filename)
        try:
            if(state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]):
                print("Best Model details:\n","valid_m",str(state['valid_score_m']), "test_m",str(state["test_score_m"]),
                                          "valid",str(state['valid_score_e2']), "test",str(state["test_score_e2"]),
                                          "valid_e1",str(state['valid_score_e1']),"test_e1",str(state["test_score_e1"]))
                best_name = os.path.join(self.save_directory, "best_valid_model.pt")
                self.best_mrr_on_valid = {"valid_m":state['valid_score_m'], "test_m":state["test_score_m"], 
                                          "valid":state['valid_score_e2'], "test":state["test_score_e2"],
                                          "valid_e1":state['valid_score_e1'], "test_e1":state["test_score_e1"]}

                if(os.path.exists(best_name)):
                    os.remove(best_name)
                torch.save(state, best_name)#os.symlink(os.path.realpath(filename), best_name)
        except:
            utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file)
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))
        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current '+
                                          'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

    def start(self, steps=50, batch_count=(20, 10), mb_start=0):
        start = time.time()
        losses = []
        count = 0
        for i in range(mb_start, steps):
            l, reg, debug = self.step()
            losses.append(l)
            suffix = ("| Current Loss %8.4f | "%l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                         (numpy.mean(losses))
            suffix += "reg %6.3f | time %6.0f ||"%(reg, time.time()-start)
            suffix += debug
            prefix = "Mini Batches %5d or %5.1f epochs"%(i+1, i*self.batch_size/self.train.kb.facts.shape[0])
            utils.print_progress_bar(len(losses), batch_count[0],prefix=prefix, suffix=suffix)
            if len(losses) >= batch_count[0]:
                losses = []
                count += 1
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks)
                    test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks)
                    self.scoring_function.train()
                    count = 0
                    print()
                    self.save_state(i, valid_score, test_score)
        print()
        print("Ending")
        print(self.best_mrr_on_valid["valid"])
        print(self.best_mrr_on_valid["test"])

