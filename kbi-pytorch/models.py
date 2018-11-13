import torch
import utils


class distmult(torch.nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(distmult, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

    def forward(self, s, r, o):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r = self.R(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)
        return (s*r*o).sum(dim=-1)

    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        s = self.E(s)
        r = self.R(r)
        o = self.E(o)
        if self.reg==2:
            return (s*s+o*o+r*r).sum()
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if (not self.unit_reg and not self.display_norms):
            return ""
        e_norms = self.E.weight.data.norm(2, dim=-1)
        r_norms = self.R.weight.data.norm(2, dim=-1)
        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r, min_r = torch.max(r_norms), torch.min(r_norms)
        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
        if self.display_norms:
            return "E[%4f, %4f] R[%4f, %4f]" % (max_e, min_e, max_r, min_r)
        else:
            return ""


class complex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None):
        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

    def forward(self, s, r, o):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)
        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        result = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        return result.sum(dim=-1)

    def regularizer(self, s, r, o):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        return (s_re*s_re+o_re*o_re+r_re*r_re+s_im*s_im+r_im*r_im+o_im*o_im).sum()

    def post_epoch(self):
        return ""


class adder_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, model1_name, model1_arguments, model2_name, model2_arguments):
        super(adder_model, self).__init__()

        model1 = globals()[model1_name]
        model1_arguments['entity_count'] = entity_count
        model1_arguments['relation_count'] = relation_count
        model2 = globals()[model2_name]
        model2_arguments['entity_count'] = entity_count
        model2_arguments['relation_count'] = relation_count

        self.model1 = model1(**model1_arguments)
        self.model2 = model2(**model2_arguments)
        self.minimum_value = self.model1.minimum_value + self.model2.minimum_value

    def forward(self, s, r, o):
        return self.model1(s, r, o) + self.model2(s, r, o)

    def post_epoch(self):
        return self.model1.post_epoch()+self.model2.post_epoch()

    def regularizer(self, s, r, o):
        return self.model1.regularizer(s, r, o) + self.model2.regularizer(s, r, o)


class typed_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0):
        super(typed_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

    def forward(self, s, r, o):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)
        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o):
        return self.base_model.regularizer(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class DME(torch.nn.Module):
    """
    DM+E model.
    deprecated. Use Adder model with DM and E as sub models for more control
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False):
        super(DME, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E_DM = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_DM = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        torch.nn.init.normal_(self.E_DM.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_DM.weight.data, 0, 0.05)

        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tail.weight.data, 0, 0.05)

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.diplay_norms=display_norms

    def forward(self, s, r, o):
        s_DM = self.E_DM(s) if s is not None else self.E_DM.weight.unsqueeze(0)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o) if o is not None else self.E_DM.weight.unsqueeze(0)

        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)

            s_DM.data.clamp_(-self.clamp_v, self.clamp_v)
            r_DM.data.clamp_(-self.clamp_v, self.clamp_v)

        out = (s*r_head+o*r_tail).sum(dim=-1) + (s_DM*r_DM*o_DM).sum(dim=-1)
        return out

    def regularizer(self, s, r, o):
        s_DM = self.E_DM(s)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o)

        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)

        return (s*s+o*o+r_head*r_head+r_tail*r_tail+s_DM*s_DM+r_DM*r_DM+o_DM*o_DM).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        e_DM_norms = self.E_DM.weight.data.norm(2, dim=-1, keepdim=True)
        r_DM_norms = self.R_DM.weight.data.norm(2, dim=-1)

        max_e_DM, min_e_DM = torch.max(e_DM_norms), torch.min(e_DM_norms)
        max_r_DM, min_r_DM = torch.max(r_DM_norms), torch.min(r_DM_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
            self.E_DM.weight.data.div_(e_DM_norms)
        if self.diplay_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail, max_e_DM, min_e_DM, max_r_DM, min_r_DM)
        else:
            return ""


class E(torch.nn.Module):
    """
    E model \n
    scoring function (s, r, o) = s*r_h + o*r_o
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=True, clamp_v=None, display_norms=False):
        super(E, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tail.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.display_norms = display_norms

    def forward(self, s, r, o):
        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)
        return (s*r_head+o*r_tail).sum(dim=-1)

    def regularizer(self, s, r, o):
        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)
        return (s*s+o*o+r_head*r_head+r_tail*r_tail).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
        if self.display_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail)
        else:
            return ""


