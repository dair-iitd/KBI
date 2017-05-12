# Joint Matrix-Tensor Factorization for Knowledge Base Inference
Code for our TACL 2017 Submission titled "Joint Matrix-Tensor Factorization for Knowledge Base Inference"

Joint Matrix-Tensor Factorization for Knowledge Base Inference, Prachi Jain, Shikhar Murty, Mausam, Soumen Chakrabarti 


# Running instructions
First clone the repository:
```
git clone https://github.com/MurtyShikhar/KBI.git
```

Make sure you have Keras by running:
```
sudo pip install keras
```
Also, ensure you have the latest versions of both **Theano** and **numpy**.


To run the atomic models(where every relation is considered as an atom), for example DistMult, E or TransE, run the following:

```
<PRACHI: Please put command here>
```
where MODEL can be **distMult**, **E** , **complex** , **TransE**, DATASET is **wn18**, **fb15k**, **fb15k-237**, **nyt-fb** and OPTIMIZER is
either **Adagrad** or **RMSprop**

To make the train data again (in the form of matrices), run
```
dump_data.py 
```

and  for creating data for MF, run 
'''
dump_data_pairwise.py 
'''

with the right set of parameters. (For info on parameters, run dump_data.py -h)



