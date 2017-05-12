dataset="NYT_FB"
if [[ $dataset == "WN18" ]]; then
    ent=40943
    rels=18
    ent_pairs=151120
elif [[ $dataset == "FB15k" ]]; then
    ent=14951
    rels=1345
    ent_pairs=467266
elif [[ $dataset == "NYT_FB" ]]; then
    ent=24526
    rels=4111
    ent_pairs=41857
fi


gpu_id=0
neg_samples=401
training=1
batch_size=4096
embedding_dimension=100
num_epochs=200
model='MF'
l2=0.0
eval_after=15
eval_every=10

