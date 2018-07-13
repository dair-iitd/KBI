# Knowledge Base Inference
This project contains Pytorch and Keras(theano) implementation of Knowledge Base Inference (KBI) Models used in the following publications [1][2]. The code has been developed at Indian Institute of Technology, Delhi (IIT Delhi). The KBI model in this repository traines over structured knowledge bases like FB15k, WN18, NYT-FB and Yago3-10. You can also add your own KB seamlessly. 

[1] Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference. Jain, Prachi and Murty, Shikhar and ., Mausam and Chakarbarti, Soumen. IJCAI 2018
[2] Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision. Jain, Prachi and Kumar, Pankaj and ., Mausam and Chakarbarti, Soumen. ACL 2018

## Repository Navigation:
./kbi-keras: Keras (theano) code for our IJCAI 2018 submission titled "Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference" and ACL 2018 submission titled "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision"

This folder has code for the following models: DM, Complex, E, TransE, F+E(AS), F+Complex(AS), F+E+DM(AS), F+Complex (AL), F+Complex(RAL), TypedDM, TypedComplex  

./kbi-pytorch: Pytorch code for our ACL 2018 submission titled "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision"

This folder has code for the following models: DM, Complex, TypedDM, TypedComplex. 

## Datasets
For the datasets, please contact the authors. 

## Model performance summary
The following table reports the best scores obtained by each of the model on FB15k. For results on more datasets, please check the README inside code folder, you may also refer to the papers. Note that, we have implemented some models in both theano and keras, and both have different performance. We report the best scores here.

| Model | -- | FB15K |--  |
| -----|-- |---|--|
| -----| MRR | H10| H1| 
| DM | 69.76 | 86.80 | 59.34 | 
| TypeDM | 75.39 | 89.26 | 66.09 |
| Complex | 68.50 | 87.08 | 57.04 | 
| TypeComplex | 75.43 | 89.45 | 66.18 | 
|E||||
|TransE||||
|F|20.21|16.26|27.42|
|F+Complex(AS)|16.84|11.48|26.42|
|F+Complex(RAL)|67.46|56.00|85.80|

## License
This software comes under a non-commercial use license, please see the LICENSE file.

## Cite Us
If you use either of the implementations (keras(theano)/pytorch). Please cite either or both the paper. See BibteX below:
```
@inproceedings{jain2018type,
	title = {{Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision}},
	author = {Jain, Prachi and Kumar, Pankaj and ., Mausam and Chakarbarti, Soumen},
	booktitle = {ACL},
	year = {2018}
}

@inproceedings{jain2018joint,
	title = {{Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference}},
	author = {Jain, Prachi and Murty, Shikhar and ., Mausam and Chakarbarti, Soumen},
	booktitle = {IJCAI},
	year = {2018}
}
```
 
