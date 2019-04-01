
The pytorch implementation for the knowledge baase completion models presented (including the baselines) in the paper "Type-Sensitive Knowledge Base Inference Without Explicit Type Supervision" at ACL2018.

# Requirements
```
python >= 3.5.2
pytorch >= 0.4.0
sklearn >= 0.19.0
```

# Running Instructions

After cloning the repository, check if all the above requirements are met. Also, download the datasets if you need. The script to run is
```
main.py [-h] -d DATASET -m MODEL -a MODEL_ARGUMENTS [-o OPTIMIZER] -l
               LOSS -r LEARNING_RATE -g REGULARIZATION_COEFFICIENT
               [-c GRADIENT_CLIP] [-e MAX_EPOCHS] [-b BATCH_SIZE]
               [-x EVAL_EVERY_X_MINI_BATCHES] [-y EVAL_BATCH_SIZE]
               [-n NEGATIVE_SAMPLE_COUNT] [-s SAVE_DIR] [-u RESUME_FROM_SAVE]
               [-v OOV_ENTITY] [-q VERBOSE] [-z DEBUG] [-k HOOKS]
               [--data_repository_root DATA_REPOSITORY_ROOT]
```

Here, __DATASET__ is the name of the dataset (the train, test and valid files should be present in a folder of this name under the __DATA_REPOSITORY_ROOT__ directory, which is by default "data" in the same directory) and __MODEL__ is the name of the model (examples of already implemented models are distmult, complex and typed_model) (These names are derived automatically from the class definitions provided in models.py file). __MODEL_ARGUMENTS__ is a json string that contains all the model parameters, including model dimensions and the base model in case of typed models (for a more detailed description of the model parameters, see the model's \__init\__ function, keeping in mind that the parameters entity_count and relation_count are filled from the dataset). LOSS is the loss function (softmax_loss and logistic_loss implemented in losses.py). Here are some sample runs of the models (require at least 11GB GPU RAM)

```
python3 main.py -d fb15k -m distmult -a '{"embedding_dim":200, "clamp_v":1.0}' -l logistic_loss -r 0.5 -g 0.01 -b 4000 -x 2000 -n 10 -v 1 -q 1

python3 main.py -d fb15k -m complex -a '{"embedding_dim":100, "clamp_v":1.0}' -l logistic_loss -r 0.5 -g 0.01 -b 4000 -x 2000 -n 10 -v 1 -q 1

python3 main.py -d fb15k -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -r 0.5 -g 0.3 -b 4000 -x 2000 -n 200 -v 1 -q 1

python3 main.py -d yago -m complex -a '{"embedding_dim":100, "clamp_v":1.0}' -l logistic_loss -r 0.5 -g 0.01 -b 4000 -x 2000 -n 10 -v 1 -y 10

python3 main.py -d yago -m typed_model -a '{"embedding_dim":19, "base_model_name":"distmult", "base_model_arguments":{"embedding_dim":180, "unit_reg":1}, "mult":20.0}' -l softmax_loss -r 0.5 -g 0.30 -b 4000 -x 2000 -n 200 -v 1 -y 25 -e 500

```


# Results Table
Following are the mean MRR (<e1,r,?> and <?,r,e2>), HITS@10 and HITS@1 of respective models

| Model | -- | FB15K |--  |--| YAGO |-- |
| -----|-- |---|--|--|--|--|
| -----| MRR | H10| H1|MRR|H10|H1|
| DM | 69.76 | 86.80 | 59.34 | 58.34 | 75.92 | 48.66 |
| TypeDM | 75.39 | 89.26 | 66.09 | 58.57 | 75.00 | 50.76 |
| Complex | 68.50 | 87.08 | 57.04 | 57.55 | 75.70 | 47.50 |
| TypeComplex | *75.87* | *87.12* | *66.18* | 58.02 | 72.80 | 50.16 |
<!--- | TypeComplex | 75.43 | 89.45 | 68.23 | 58.02 | 72.80 | 50.16 | --->

To implement a new model or loss function, one just needs to create a corresponding class inheriting torch.nn.Module in the corresponding file (including some additional methods like post_epoch in case of models).
