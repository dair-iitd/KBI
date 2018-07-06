import torch
import sklearn.decomposition
import sklearn.manifold
import numpy
from mpl_toolkits.mplot3d import Axes3D
import kb
import matplotlib.pyplot

def plot_weights(w, sne=False):
    w = w.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=3)
    else:
        pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(w)
    w = pca.transform(w)
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.scatter(w[:, 0], w[:, 1], w[:, 2])
    matplotlib.pyplot.show()


def plot_weights_2d(w, sne=False, metric='cosine'):
    w = w.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=2, verbose=1, metric=metric)
    else:
        pca = sklearn.decomposition.PCA(n_components=2)
    w = pca.fit_transform(w)
    matplotlib.pyplot.scatter(w[:, 0], w[:, 1])
    matplotlib.pyplot.show()


def fb15k_type_map():
    f = open('data/fb15k/entity2type.txt')
    f = [x.split() for x in f.readlines()]
    mp = {}
    def remove_types(fl, tp):
        ls = []
        rest = []
        for x in fl:
            present = False
            for y in x[1:]:
                if y.startswith(tp):
                    present = True
            if not present:
                ls.append(x)
            else:
                rest.append(x)
        return ls, rest
    rest, people = remove_types(f, '/people')
    rest, location = remove_types(rest, '/location')
    rest, organisation = remove_types(rest, '/organisation')
    rest, film = remove_types(rest, '/film')
    rest, sports = remove_types(rest, '/sports')
    def get_set(fl):
        return set([x[0] for x in fl])
    people = get_set(people)
    location = get_set(location)
    organisation = get_set(organisation)
    film = get_set(film)
    sports = get_set(sports)
    rest = get_set(rest)
    types = [people, location, organisation, film, sports, rest]
    def get_type(e):
        for i, t in enumerate(types):
            if e in t:
                return i
        return len(types)-1
    k = kb.kb('data/fb15k/train.txt')
    for i, e in enumerate(k.entity_map):
        mp[i] = get_type(e)
    print(mp)
    return mp


def colored_plot(fb15k_weights, sne=False, td=False, dis='cosine'):
    w = fb15k_weights.cpu().numpy()
    if sne:
        pca = sklearn.manifold.TSNE(n_components=3 if td else 2, metric=dis, verbose=2)
    else:
        pca = sklearn.decomposition.PCA(n_components=3 if td else 2)
    tw = pca.fit_transform(w)
    return tw


def plt(tw, w, td, entity_type_map):
    col = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    fig = matplotlib.pyplot.figure()
    if(td):
        axes = fig.add_subplot(111, projection='3d')
        for i in range(w.shape[0]):
            axes.scatter(tw[i, 0], tw[i, 1], tw[i, 2], c=col[entity_type_map[i]], alpha=0.1)
    else:
        axes = matplotlib.pyplot
        for i in range(w.shape[0]):
            axes.scatter(tw[i, 0], tw[i, 1], c=col[entity_type_map[i]], alpha=0.1)
    matplotlib.pyplot.show()


def fb15k_type_map_fine():
    fl = open("data/fb15k/entity_mid_name_type_typeid.txt").readlines()
    fl = [x.strip().split('\t') for x in fl]
    result = {}
    for line in fl:
        result[line[0]] = int(line[3])
    return result


class type_mrr_hook(object):
    def __init__(self, kb, type_id):
        self.type_id = type_id
        self.inverse_rank_total = 0
        self.count = 0
        self.kb = kb
    def __call__(self, s, r, o, ranks, top_predictions, top_scores, expected_type, top_predictions_type):
        f = expected_type.eq(self.type_id).float()
        self.inverse_rank_total += (f/ranks.float()).sum()
        self.count += f.sum()
    def end(self):
        print(self.type_id, self.inverse_rank_total/self.count)
    def start(self):
        self.inverse_rank_total = 0
        self.count = 0

class rwise_mrr_hook(object):
    def __init__(self, kb, relation_count):
        self.relation_count = relation_count
        self.inverse_rank_total = torch.zeros(relation_count).cuda()
        self.count = torch.zeros(relation_count).cuda()
        self.kb = kb
    def __call__(self, s, r, o, ranks, top_predictions, top_scores, expected_type, top_predictions_type):
        self.count.scatter_add_(0, r.squeeze(), torch.FloatTensor([1.0]).cuda().expand_as(r.squeeze()))
        self.inverse_rank_total.scatter_add_(0, r.squeeze(), 1/ranks)
    def end(self):
        result = self.inverse_rank_total/self.count
    def start(self):
        self.inverse_rank_total [:] = 0
        self.count[:] = 0

def load_hooks(hooks, kb):
    result = []
    for hook_param in hooks:
        hook_class = globals()[hook_param['name']]
        hook_param['arguments']['kb'] = kb
        result.append(hook_class(**hook_param['arguments']))
    return result

