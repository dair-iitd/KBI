#1_dataset 2_model 3_logFile 4_type_combined/atomic 5_train_0/1 6_l2_entity_pair 7_gpu_id 8_num_epochs 9_learningRate 10_unit_norm_reg 11_evalDev 12_vect_dim 13_theo_reg 14_type_dim 
#!/bin/bash

echo $# arguments 

init_model=0 #default is 0

learning_rate=0.5
batch_size=2000
unit_norm_reg=0
gpu_id=0
l2_entity_pair=0
num_epochs=200
evalDev=0
vect_dim=200

type_dim=${14}

theo_reg=${13}
l2=0.0
l2_DM=0.0
l2=0.0

neg_samples=10 #5 #200 for wn18 #5 - for fb15k theo

eval_after=10 #50 
eval_every=10 #50 

oov_train=1 
oov_avg=0

oov_eval=1

loss="logistic" #"mm" #ll"


model_path="complex_fb15k_dim200_tmp.h5 " 

if [[ $# == 3 ]]; then 
    echo "By default run file is train combined models and training is set to false"
    run_file="train_combined_model.py"
    train=0
elif [[ $# == 4 ]]; then
    echo "By default training is disabled"
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=0
elif [[ $# == 5 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
elif [[ $# == 7 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
elif [[ $# == 8 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
elif [[ $# == 9 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
elif [[ $# == 10 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
    unit_norm_reg=${10}
elif [[ $# == 11 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
    unit_norm_reg=${10}
    evalDev=${11}
elif [[ $# == 12 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
    unit_norm_reg=${10}
    evalDev=${11}
    vect_dim=${12}

elif [[ $# == 13 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
    unit_norm_reg=${10}
    evalDev=${11}
    vect_dim=${12}
    theo_reg=${13}

elif [[ $# == 14 ]]; then
    if [[ $4 == "atomic" ]]; then
        run_file="trainAtomicModels.py"
    elif [[ $4 == "combined" ]]; then
        run_file="train_combined_model.py"
    else
        echo "You passed an illegal execution mode"
    fi
    train=$5
    l2_entity_pair=$6
    gpu_id=$7
    num_epochs=$8
    learning_rate=$9
    unit_norm_reg=${10}
    evalDev=${11}
    vect_dim=${12}
    theo_reg=${13}
    type_dim=${14} 
fi


dataset=$1
model=$2
if [[ $dataset == "wn18" ]]; then
    ent=40943
    file_dir="original/encoded_data/without-text"
    rels=18
    ent_pairs=151120
elif [[ $dataset == "fb15k" ]]; then
    ent=14951
    file_dir="original/encoded_data/without-text"
    rels=1345
    ent_pairs=467266
elif [[ $dataset == "nyt-fb" ]]; then
    ent=24526
    file_dir="re-split/encoded_data/with-text"
    rels=4111
    ent_pairs=41857
elif [[ $dataset == "fb15k-237" ]]; then
    ent=14541
    file_dir="original/encoded_data/without-text"
    rels=237
    ent_pairs=283868
elif [[ $dataset == "fb15k-237-t" ]]; then
    ent=14541
    file_dir="original/encoded_data/with-text"
    rels=2740640
    ent_pairs=2042193
    dataset="fb15k-237"
elif [[ $dataset == "yago" ]]; then
    file_dir="encoded_data/without-text"
    dataset="yago3-10"
    ent=123182
    rels=37
    ent_pairs=797297
fi

log_file=$3

echo Executing:
echo Running $run_file !!!!!!!!
cmd="THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $l2_DM -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train -oov_avg $oov_avg -oov_train $oov_train -unit_norm $unit_norm_reg -rate $learning_rate -vect_dim $vect_dim -l2_entity_pair $l2_entity_pair -evalDev $evalDev -oov_eval $oov_eval -loss $loss -init_model $init_model -theo_reg $theo_reg -type_dim $type_dim -log_file $log_file"
echo $cmd

THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $l2_DM -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train -oov_avg $oov_avg -oov_train $oov_train -unit_norm $unit_norm_reg -rate $learning_rate -vect_dim $vect_dim -l2_entity_pair $l2_entity_pair -evalDev $evalDev -oov_eval $oov_eval -loss $loss -init_model $init_model -theo_reg $theo_reg -type_dim $type_dim -log_file $log_file

