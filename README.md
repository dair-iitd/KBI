# Knowledge Base Inference
This project contains Pytorch and Keras(theano) implementation of Knowledge Base Inference (KBI) Models used in the following publications [1][2]. The code has been developed at Indian Institute of Technology, Delhi (IIT Delhi). The KBI model in this repository traines over structured knowledge bases like FB15k, WN18, NYT-FB and Yago3-10. You can also add your own KB seamlessly. 

[1] Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference. Jain, Prachi and Murty, Shikhar and ., Mausam and Chakarbarti, Soumen. IJCAI 2018
[2] Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision. Jain, Prachi and Kumar, Pankaj and ., Mausam and Chakarbarti, Soumen. ACL 2018

## Repository Navigation:
./kbi-keras: Keras (theano) code for our IJCAI 2018 submission titled "Mitigating the Effect of Out-of-Vocabulary Entity Pairs in Matrix Factorization for KB Inference" and ACL 2018 submission titled "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision"

./kbi-pytorch: Pytorch code for our ACL 2018 submission titled "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision" 

## Datasets
For the datasets, please contact the authors. 

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
 
