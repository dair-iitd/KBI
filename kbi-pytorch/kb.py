import numpy
import torch


class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """
    def __init__(self, filename, em=None, rm=None, add_unknowns=True):
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entites are to be acknowledged or put as <UNK> token.
        """
        self.entity_map = {} if em is None else em
        self.relation_map = {} if rm is None else rm
        if filename is None:
            return
        facts = []
        with open(filename) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]
            for l in lines:
                if(add_unknowns):
                    if(l[0] not in self.entity_map):
                        self.entity_map[l[0]] = len(self.entity_map)
                    if(l[2] not in self.entity_map):
                        self.entity_map[l[2]] = len(self.entity_map)
                    if(l[1] not in self.relation_map):
                        self.relation_map[l[1]] = len(self.relation_map)
                facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                        len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)])
        self.facts = numpy.array(facts, dtype='int64')

    def augment_type_information(self, mapping):
        """
        Augments the current knowledge base with entity type information for more detailed evaluation.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """
        self.type_map = mapping
        entity_type_matrix = numpy.zeros((len(self.entity_map), 1))
        for x in self.type_map:
            entity_type_matrix[self.entity_map[x], 0] = self.type_map[x]
        entity_type_matrix = torch.from_numpy(numpy.array(entity_type_matrix))
        self.entity_type_matrix = entity_type_matrix

    def compute_degree(self, out=True):
        """
        Computes the in-degree or out-degree of relations\n
        :param out: Whether to compute out-degree or in-degree
        :return: A numpy array with the degree of ith ralation at ith palce.
        """
        entities = [set() for x in self.relation_map]
        index = 2 if out else 0
        for i in range(self.facts.shape[0]):
            entities[self.facts[i][1]].add(self.facts[i][index])
        return numpy.array([len(x) for x in entities])
        

def union(kb_list):
    """
    Computes a union of multiple knowledge bases\n
    :param kb_list: A list of kb
    :return: The union of all kb in kb_list
    """
    l = [k.facts for k in kb_list]
    k = kb(None, kb_list[0].entity_map, kb_list[0].relation_map)
    k.facts = numpy.concatenate(l, 0)
    return k


def dump_mappings(mapping, filename):
    """
    Stores the mapping into a file\n
    :param mapping: The mapping to store
    :param filename: The file name
    :return: None
    """
    data = [[x, mapping[x]] for x in mapping]
    numpy.savetxt(filename, data)


def dump_kb_mappings(kb, kb_name):
    """
    Dumps the entity and relation mapping in a kb\n
    :param kb: The kb
    :param kb_name: The fine name under which the mappings should be stored.
    :return:
    """
    dump_mappings(kb.entity_map, kb_name+".entity")
    dump_mappings(kb.relation_map, kb_name+".relation")



