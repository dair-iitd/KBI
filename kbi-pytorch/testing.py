import kb
import models
import losses
import numpy
import torch


ne = 3
nr = 2
nf = 4


et = numpy.array([[1.0, 2.0], [0, 1], [-1, 2]])
rht = numpy.array([[-1.0, 1.5], [0, 1]])
rtt = numpy.array([[1.0, -2.0], [2,2]])
e = numpy.array([[1.0, 2.0,3,4,5], [3,2,1,0,-1], [3,4,2,1,3]])
r = numpy.array([[3,1,2,4.0,5], [3,2,3,2,3]])



scoring_function = models.typed_distmult(ne, nr, 5, 2)

print(scoring_function.E_t.weight)
print(scoring_function.R_ht.weight)
print(scoring_function.R_tt.weight)
print(scoring_function.base_model.E.weight)
print(scoring_function.base_model.R.weight)


scoring_function.E_t.weight.data = torch.from_numpy(et)
scoring_function.R_ht.weight.data = torch.from_numpy(rht)
scoring_function.R_tt.weight.data = torch.from_numpy(rtt)
scoring_function.base_model.E.weight.data = torch.from_numpy(e)
scoring_function.base_model.R.weight.data = torch.from_numpy(r)

print(scoring_function.E_t.weight)
print(scoring_function.R_ht.weight)
print(scoring_function.R_tt.weight)
print(scoring_function.base_model.E.weight)
print(scoring_function.base_model.R.weight)




ll = losses.softmax_loss()

s = torch.autograd.Variable(torch.from_numpy(numpy.array([[1], [0], [2]])))
r = torch.autograd.Variable(torch.from_numpy(numpy.array([[1], [0], [1]])))
o = torch.autograd.Variable(torch.from_numpy(numpy.array([[1], [0], [0]])))

ns = torch.autograd.Variable(torch.from_numpy(numpy.array([[0, 1], [1, 2], [0, 1]])))
no = torch.autograd.Variable(torch.from_numpy(numpy.array([[0, 1], [1, 2], [0, 1]])))

def fun(s, r, o, ns, no):
	fp = scoring_function(s, r, o)
	fns = scoring_function(ns, r, o)
	fno = scoring_function(s, r, no)
	print(fp, fns, fno)

	loss = ll(fp, fns, fno)

	print(loss)

fun(s, r, o, ns, no)