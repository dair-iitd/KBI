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
bash run.sh <DATASET> distMult <LOG-FILE> atomic 1 0 0 100 0.5 0 0 100 0.01 0

Complex
bash run.sh <DATASET> distMult <LOG-FILE> atomic 1 0 0 100 0.5 0 0 100 0.01 0

E
bash run.sh <DATASET> E <LOG-FILE> atomic 1 0 0 100 0.5 1 0 100 0 0
Don't forget to change the loss to ll in run.sh

TransE 
bash run.sh <DATASET> TransE <LOG-FILE> atomic 1 0 0 100 0.5 1 0 100 0 0
Don't forget to change the loss to mm in run.sh

F+E(AS)
F+Complex(AS)
F+E+DM(AS)
F+Complex (AL) 
F+Complex(RAL)
bash run_combined.sh <DATASET> adderNet <LOG-FILE> combined 1 0 0 100 0.5 0 0 0 1 100 0

TypedDM
bash run.sh <DATASET> typedDM <LOG-FILE> atomic 1 0 0 180 0.5 0 0 100 0.01 19
Don't forget to change the loss to in run.sh



TypedComplex
<to be added>

```


where DATASET is **wn18**, **fb15k**, **fb15k-237**, **nyt-fb** and OPTIMIZER is either **Adagrad** or **RMSprop**

### Model performance summary
We report the scores obtained from the models trained using keras(theano) code (100 dim model) here.

| Model | -- | FB15K |--  |
| -----|-- |---|--|
| -----| MRR | H10| H1|
| DM | 60.82 | 46.51 | 84.78 |
| Complex | 66.97 | 55.21 | 85.60 |
|E|22.86|16.4|35.04|
|TransE|43.11|24.99|71.97|
|F|20.21|16.26|27.42|
|F+Complex(AS)|16.84|11.48|26.42|
|F+Complex(RAL)|67.46|56.00|85.80|


### Stored models
Folder ./bestModels stores the models trained from the code you run.
