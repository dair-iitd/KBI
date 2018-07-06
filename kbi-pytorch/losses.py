import torch


class softmax_loss(torch.nn.Module):
    def __init__(self):
        super(softmax_loss, self).__init__()
    def forward(self, positive, negative_1, negative_2):
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        den_e1 = (negative_1-max_den_e1).exp().sum(dim=-1, keepdim=True)
        den_e2 = (negative_2-max_den_e2).exp().sum(dim=-1, keepdim=True)
        losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        return -losses.mean()


class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()
    def forward(self, positive, negative_1, negative_2):
        scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        x = torch.log(1+torch.exp(-scores*truth))
        total = x.sum()
        return total/((positive.shape[1]+negative_1.shape[1]+negative_2.shape[1])*positive.shape[0])

