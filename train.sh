
Train DSMIL
python /system/user/publicwork/yitaocai/Master_Thesis/train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT_VIG --num_classes 3 --epochs 50 --n 3 --split determined --df df_combined --in_size 2000




python /system/user/publicwork/yitaocai/Master_Thesis/train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3 --epochs 50 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000


Train attentionbase
python /system/user/publicwork/yitaocai/Master_Thesis/code/train_AttentionBase.py --seed 1337 --device cuda:3  --num_classes 2 --in_size 1000 --dataset CNT --epoch 50

python /system/user/publicwork/yitaocai/Master_Thesis/code/train_AttentionBase.py --seed 1337 --device cuda:2 --dataset CNT --num_classes 3 --epochs 50 --split random --df df_combined --weighted_sampling False --in_size 1000


Tune TRANSFORMER
python /system/user/publicwork/yitaocai/Master_Thesis/tune_topk_transformer.py --num_classes 2

Tune Topk MultiheadAttention
python /system/user/publicwork/yitaocai/Master_Thesis/tune_topk_multiheadattention.py --num_classes 2



Train TRANSFORMER

python /system/user/publicwork/yitaocai/Master_Thesis/code/train_topk_transformer.py  --seed 1337 --device cuda:0 --dataset CNT --num_classes 2 --epochs 50  --split random --df df_combined --in_size 1000 --topk 8 --num_heads 12 --num_layers 2 --head_dim 128 --embed_x_dim 128 --critical_features_from embedding --classification maxpool1d

python /system/user/publicwork/yitaocai/Master_Thesis/train_topk_transformer.py --seed 1337 --device cuda:3 --dataset CNT --num_classes 3 --epochs 30 --n 2 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000 --topk 3 --num_heads 2 --num_layers 5 --classification LPPool1d

Train Topk MultiheadAttention
python /system/user/publicwork/yitaocai/Master_Thesis/code/train_TopkMultiheadAttention.py --seed 1337 --device cuda:0 --dataset CNT --num_classes 2 --epochs 50 --split determined --df df_combined --in_size 1000 --num_heads 64 --topk 5 --head_dim 128 --head_proj conv1d_ckernels --embed_x_dim 256 --critical_features_from embedding

python /system/user/publicwork/yitaocai/Master_Thesis/train_TopkMultiheadAttention.py --seed 1337 --device cuda: --dataset CNT --num_classes 3 --epochs 30 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000 --num_heads 2 --topk 2


python /system/user/publicwork/yitaocai/Master_Thesis/train_DSMIL.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3 --epochs 50 --n 1 --split random --df df_combined --weighted_sampling True --batch_size 420 --w0 33 --w1 83 --w2 65 --in_size 1000

TEST


python /system/user/publicwork/yitaocai/Master_Thesis/test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3  --model_path /system/user/publicwork/yitaocai/Master_Thesis/model/model_3C_CNT/3C_n5_seed88_SGD_random_12262022.pt --df df_combined --in_size 1000
python /system/user/publicwork/yitaocai/Master_Thesis/test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 3  --model_path image.png --df df_combined --in_size 1000

python /system/user/publicwork/yitaocai/Master_Thesis/test.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2  --model_path /system/user/publicwork/yitaocai/Master_Thesis/model/model_2C/2C_n5_seed8_SGD_determined_12212022.pt --df df_combined --in_size 1000

python /system/user/publicwork/yitaocai/Master_Thesis/code/test_TopkMultiHeadAttention.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2 --df df_combined --in_size 1000 --num_heads 32 --topk 5 --head_dim 128 --head_proj conv1d_ckernels --embed_x_dim 256 --critical_features_from embedding --model_path /system/user/publicwork/yitaocai/Master_Thesis/model/topk_multihead_2C/Topk5_32heads_conv1d_ckernels_CFfromembedding_2C_CNT_weightedFalse_determined_04152023.pt


python /system/user/publicwork/yitaocai/Master_Thesis/test_TopkCFTransformer.py --seed 1337 --device cuda:1 --dataset CNT --num_classes 2 --df df_combined --in_size 1000 --num_heads 2 --topk 5 --num_layers 3 --classification maxpool1d --model_path /system/user/publicwork/yitaocai/Master_Thesis/model/topk_transformer/Topk5_2heads_3layers_2C_CNT_weightedFalse_random_02282023.pt


python /system/user/publicwork/yitaocai/Master_Thesis/code/test_AttentionBase.py --seed 1337 --device cuda:3 --dataset CNT --num_classes 3 --df df_combined --in_size 1000  --model_path /system/user/publicwork/yitaocai/Master_Thesis/model/AttentionBase/3C_CNT_n1_weightedTrue_random_03302023.pt



when tmux lost server connection: rm -r /tmp/tmux-`id -u`

attention_map_TopkMultihead_2C