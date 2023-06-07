
Train DSMIL
python train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT_VIG --num_classes 3 --epochs 50 --n 3 --split determined --df df_combined --in_size 2000




python train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3 --epochs 50 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000


Train attentionbase
python train_AttentionBase.py --seed 1337 --device cuda:3  --num_classes 2 --in_size 1000 --dataset CNT --epoch 50

python train_AttentionBase.py --seed 1337 --device cuda:2 --dataset CNT --num_classes 3 --epochs 50 --split random --df df_combined --weighted_sampling False --in_size 1000


Tune TRANSFORMER
python tune_topk_transformer.py --num_classes 2

Tune Topk MultiheadAttention
python tune_topk_multiheadattention.py --num_classes 2



Train TRANSFORMER

python train_topk_transformer.py  --seed 1337 --device cuda:0 --dataset CNT --num_classes 2 --epochs 50  --split random --df df_combined --in_size 1000 --topk 8 --num_heads 12 --num_layers 2 --head_dim 128 --embed_x_dim 128 --critical_features_from embedding --classification maxpool1d

python train_topk_transformer.py --seed 1337 --device cuda:3 --dataset CNT --num_classes 3 --epochs 30 --n 2 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000 --topk 3 --num_heads 2 --num_layers 5 --classification LPPool1d

Train Topk MultiheadAttention
python train_TopkMultiheadAttention.py --seed 1337 --device cuda:0 --dataset CNT --num_classes 2 --epochs 50 --split determined --df df_combined --in_size 1000 --num_heads 64 --topk 5 --head_dim 128 --head_proj conv1d_ckernels --embed_x_dim 256 --critical_features_from embedding

python train_TopkMultiheadAttention.py --seed 1337 --device cuda: --dataset CNT --num_classes 3 --epochs 30 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000 --num_heads 2 --topk 2


python train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3 --epochs 50 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000

TEST


python test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3  --model_path model_3C_CNT/3C_n5_seed88_SGD_random_12262022.pt --df df_combined --in_size 1000
python test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3  --model_path image.png --df df_combined --in_size 1000

python test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2  --model_path 2C_n5_seed8_SGD_determined_12212022.pt --df df_combined --in_size 1000

python test_TopkMultiHeadAttention.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2 --df df_combined --in_size 1000 --num_heads 32 --topk 5 --head_dim 128 --head_proj conv1d_ckernels --embed_x_dim 256 --critical_features_from embedding --model_path Topk5_32heads_conv1d_ckernels_CFfromembedding_2C_CNT_weightedFalse_determined_04152023.pt


python test_TopkCFTransformer.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2 --df df_combined --in_size 1000 --num_heads 2 --topk 5 --num_layers 3 --classification maxpool1d --model_path Topk5_2heads_3layers_2C_CNT_weightedFalse_random_02282023.pt


python test_AttentionBase.py --seed 1337 --device cuda:3 --dataset CNT --num_classes 3 --df df_combined --in_size 1000  --model_path 3C_CNT_n1_weightedTrue_random_03302023.pt



