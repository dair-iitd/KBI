# Joint Matrix-Tensor Factorization for Knowledge Base Inference
kbi-keras: Code for our IJCAI 2018 submission titled "Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference" ([arXiv2017](https://arxiv.org/pdf/1706.00637.pdf))

kbi-pytorch: Code for our ACL 2018 submission titled "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision" 


# Running instructions
To run the atomic models(where every relation is considered as an atom), for example DistMult, E or TransE, run the following:

```
./run.sh <dataset> <model> atomic
```
where MODEL can be **distMult**, **E** , **complex** , **TransE**, DATASET is **wn18**, **fb15k**, **fb15k-237**, **nyt-fb** and OPTIMIZER is
either **Adagrad** or **RMSprop**


To create training data, run
```
dump_data.py 
```

and  for creating data for MF, run 
```
dump_data_pairwise.py 
```

with the right set of parameters. (For info on parameters, run dump_data.py -h)

For the datasets, please contact the authors. 


