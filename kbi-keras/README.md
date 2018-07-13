# Joint Matrix-Tensor Factorization for Knowledge Base Inference

Code for our IJCAI 2018 Submission titled [Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference.](http://www.cse.iitd.ernet.in/~mausam/papers/ijcai18b.pdf) Prachi Jain*, Shikhar Murty*, Mausam, Soumen Chakrabarti  (*Equal Contribution)

## Requirements
```
python = 2.7
keras = 2.0.2
theano
joblib
numpy
```

## Running instructions
First clone the repository:
```
https://github.com/dair-iitd/KBI.git
```
You only need code in kbi-keras/ folder. You may ignore other folders.

Make sure you have Keras by running:
```
sudo pip install keras
```
Also, ensure you have the latest versions of both **Theano** and **numpy**.

### Preparing training data
To create training data, go to folder ./prepare-data/ and run
```
python dump_data.py 
```

and for creating data for MF, run
```
python dump_data_pairwise.py 
```
with the right set of parameters. (For info on parameters, run dump_data.py -h)

This will generate required files to run the code.


### Training the model
Sample commands to train models:
```
This folder has code for the following models: 
DM
Complex
E
TransE 
F+E(AS)
F+Complex(AS)
F+E+DM(AS)
F+Complex (AL) 
F+Complex(RAL)
TypedDM
TypedComplex

```


where DATASET is **wn18**, **fb15k**, **fb15k-237**, **nyt-fb** and OPTIMIZER is either **Adagrad** or **RMSprop**

### Dataset
For the datasets, please contact the authors. 

### Stored models

