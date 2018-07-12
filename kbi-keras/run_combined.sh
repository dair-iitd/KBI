#1_dataset 2_model 3_logFile 4_type_combined/atomic 5_train_0/1 6_l2_entity_pair 7_gpu_id 8_num_epochs 9_learningRate 10_unit_norm_reg 11_shared_r 12_evalDev 13_add_loss0/1 14_vect_dim 15_theo_reg 
#!/bin/bash
echo $# arguments 

#NO change 
static_alpha=0 #if this is on - AS model MF component is multiplied with alphaMF
alphaMF=1
#No change

init_model=0 #default is 0

learning_rate=0.5
batch_size=2000
unit_norm_reg=0

gpu_id=0
num_epochs=200
add_loss=0 #if off AS model is used
evalDev=0
vect_dim=200

l2_relation_DM=0.0 #reg for rel embeddings of MF/TF models 
l2_relation_MF=${6} #${10} #$6 #0 #wt for theo reg penalty on MF
theo_reg=${20}
l2_entity=0 #reg for entity/ep in DM & MF in atomic model && reg for DM in combined 
l2_entity_pair=0 #to regularize Mf ep embeddings in combined- usually off

neg_samples=10 #1 for wn18 #5 - for fb15k theo

eval_after=50 
eval_every=50

oov_train=1 
oov_avg=0
oov_eval=1
shared_r=0 # share embedding of r b/w TF and MF model

loss="logistic" #"mm" #ll"

model_path="complex_fb15k_dim200_tmp.h5"

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
    shared_r=${11}
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
    shared_r=${11}
    evalDev=${12}
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
    shared_r=${11}
    evalDev=${12}
    add_loss=${13}
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
    shared_r=${11}
    evalDev=${12}
    add_loss=${13}
    vect_dim=${14}
elif [[ $# == 15 ]]; then
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
    shared_r=${11}
    evalDev=${12}
    add_loss=${13}
    vect_dim=${14}
    theo_reg=${15}
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
cmd="THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $l2_relation_DM -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train -oov_avg $oov_avg -oov_train $oov_train -unit_norm $unit_norm_reg -rate $learning_rate -vect_dim $vect_dim -add_tanh $add_tanh -shared_r $shared_r -norm_score $norm_score -l2_entity_pair $l2_entity_pair -static_alpha $static_alpha -alphaMF $alphaMF -static_beta $static_beta -evalDev $evalDev -l2_relation_MF $l2_relation_MF -add_loss $add_loss -oov_eval $oov_eval -loss $loss -init_model $init_model -theo_reg $theo_reg -log_file $log_file"
echo $cmd

THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $l2_relation_DM -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train -oov_avg $oov_avg -oov_train $oov_train -unit_norm $unit_norm_reg -rate $learning_rate -vect_dim $vect_dim -add_tanh $add_tanh -shared_r $shared_r -norm_score $norm_score -l2_entity_pair $l2_entity_pair -static_alpha $static_alpha -alphaMF $alphaMF -static_beta $static_beta -evalDev $evalDev -l2_relation_MF $l2_relation_MF -add_loss $add_loss -oov_eval $oov_eval -loss $loss -init_model $init_model -aux_model_loss_reg $aux_model_loss_reg -theo_reg $theo_reg -type_pair_count $type_pair_count -type_dim $type_dim -model_path $model_path &> $3
