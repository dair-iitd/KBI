#1_dataset 2_model 3_logFile 4_type_combined/atomic 5_train_0/1 6_l2_entity_pair 7_gpu_id 8_num_epochs 9_learningRate 10_static_alpha 11_alphaMF 12_static_beta 13_unit_norm_reg 14_shared_r 15_evalDev 16_add_loss0/1 17_dropout_MF 18_dropout_DM 19_vect_dim
#!/bin/bash
echo $# arguments 
learning_rate=0.5
batch_size=1000
unit_norm_reg=0
gpu_id=1
l2_entity_pair=0
num_epochs=200
add_loss=0
evalDev=1
vect_dim=200


neg_samples=50
l2=0.01
lr=0.0
eval_after=50
eval_every=10
oov_train=1
oov_avg=0
oov_eval=1

add_tanh=0
shared_r=0
norm_score=0
loss="ll"
#model_path="MF_FB15k_dim100.h5"
#model_path="distMult_1_.._DATA_REPOSITORY_wn18_original_encoded_data_without-text.h5"
#model_path="adder_fb15k-237_0.1.h5"
#model_path="adder_fb15k_0.0.h5"
#model_path="MF_FB15k_dim100.h5"
#distMult_NYT_FB_dim100.h5"
model_path="distMult_FB15k_dim100.h5"

#model_path="FE_fb15k.h5"
#model_path="distMult_FB15k-237_dim100.h5"
#"MF_FB15k-237_dim100.h5"
#"MF_FB15k_dim100.h5"

#model_path="distMult_FB15k_dim100.h5"
#"distMult_NYT_FB_dim100.h5"
#"MF_oovtrained_nyt-fb100dim.h5"
#"MF_WN18_dim100.h5"
#"distMult_wn18_dim100_maxmargin.h5"

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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}	
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}
    shared_r=${14}
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}
    shared_r=${14}
    evalDev=${15}
elif [[ $# == 16 ]]; then
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}
    shared_r=${14}
    evalDev=${15}
    add_loss=${16}
elif [[ $# == 18 ]]; then
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}
    shared_r=${14}
    evalDev=${15}
    add_loss=${16}
    dropout_MF=${17}
    dropout_DM=${18}

elif [[ $# == 19 ]]; then
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
    static_alpha=${10}
    alphaMF=${11}
    static_beta=${12}
    unit_norm_reg=${13}
    shared_r=${14}
    evalDev=${15}
    add_loss=${16}
    dropout_MF=${17}
    dropout_DM=${18}
    vect_dim=${19}
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
fi



echo Executing:
echo Running $run_file !!!!!!!!
cmd="date; time THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $lr -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train  -oov_avg $oov_avg -oov_train $oov_train -add_tanh $add_tanh -norm_score $norm_score -l2_entity_pair $l2_entity_pair -static_alpha $static_alpha -alphaMF $alphaMF -static_beta $static_beta -evalDev $evalDev -l2_relation_MF $l2_entity_pair -add_loss $add_loss -evalDev $evalDev -dropout_MF $dropout_MF -dropout_DM $dropout_DM -vect_dim $vect_dim -model_path $model_path -loss $loss" 
echo $cmd


THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpu_id,floatX=float32 python -u $run_file -neg_samples $neg_samples -num_entities $ent -num_relations $rels -num_entity_pairs $ent_pairs  -model $model -dataset ../DATA_REPOSITORY/$dataset/$file_dir -l2_entity $l2 -l2_relation $lr -batch_size $batch_size -epochs $num_epochs -eval_every $eval_every -eval_after $eval_after -train $train -oov_avg $oov_avg -oov_train $oov_train -unit_norm $unit_norm_reg -rate $learning_rate -vect_dim $vect_dim -add_tanh $add_tanh -shared_r $shared_r -norm_score $norm_score -l2_entity_pair $l2_entity_pair -static_alpha $static_alpha -alphaMF $alphaMF -static_beta $static_beta -evalDev $evalDev -l2_relation_MF $l2_entity_pair -add_loss $add_loss -dropout_MF $dropout_MF -dropout_DM $dropout_DM -oov_eval $oov_eval -loss $loss -model_path $model_path &> $3
