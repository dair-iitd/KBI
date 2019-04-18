
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
Following are the mean/<e1,r,?>/<?,r,e2> MRR, HITS@10 and HITS@1 of respective models



|     Model   | FB15k  |  --     |   --   | FB15k-237 |    --   |    --  | Yago3-10 |    --   |   --   |
|-------------|--------|---------|--------|-----------|---------|--------|----------|---------|--------|
|    --       | Mean   |   --    |  --    |      --   |  --     |  --    |    --    |   --    |    --  |
|    --       | MRR    | HITS@10 | HITS@1 | MRR       | HITS@10 | HITS@1 | MRR      | HITS@10 | HITS@1 |
| Complex     | 72.15  | 85.66   | 60.85  | 21.72     | 42.96   | 12.29  | 40.9     | 61.53   | 30.97  |
| TypeComplex | 75.93  | 86.94   | 68.45  | 25.91     | 41.14   | 18.66  |    --    |   --    |    --  |
|    --       | e1,r,? |   --    |   --   |    --     |   --    |   --   |   --     |   --    |   --   |
| Complex     | 71.83  | 87.64   | 62.18  | 26.58     | 52.11   | 15.16  | 57.39    | 74.46   | 48.26  |
| TypeComplex | 78.15  | 89.7    | 70.43  | 35.15     | 51.74   | 27.06  |    --    |   --    |    --  |
|    --       | ?,r,e2 |   --    |   --   |    --     |   --    |   --   |   --     |    --   |   --   |
| Complex     | 68.84  | 83.69   | 59.52  | 16.87     | 33.81   | 9.41   | 24.41    | 48.6    | 13.68  |
| TypeComplex | 73.71  | 84.17   | 66.48  | 16.67     | 30.55   | 10.26  |    --    |   --    |    --  |



To implement a new model or loss function, one just needs to create a corresponding class inheriting torch.nn.Module in the corresponding file (including some additional methods like post_epoch in case of models).
