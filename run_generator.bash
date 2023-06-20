# CUDA_VISIBLE_DEVICES=0,1 python generator.py \
# --expname test1 \
# --batchSize 30 --testBatchSize 1 --mag_in 15 --foolmodel resnet18-imagenette --mode train \
# --perturbation_type universal --target -1 --gpu_ids 0,1 --nEpochs 10

# CUDA_VISIBLE_DEVICES=0,1 python generator_hidden.py \
# --expname getbak_tempt \
# --batchSize 30 --testBatchSize 30 --mag_in 20 --foolmodel resnet18-imagenette --mode train \
# --perturbation_type fix_trigger --target -1 --gpu_ids 0 --nEpochs 10

# CUDA_VISIBLE_DEVICES=0 python generator_hidden_2.py \
# --expname lpips3 \
# --batchSize 12 --testBatchSize 1 --mag_in 10 --foolmodel resnet18-imagenette --mode train \
# --perturbation_type imdep --target -1 --gpu_ids 0 --nEpochs 10

CUDA_VISIBLE_DEVICES=0,1 python generator_new.py \
--expname getbak_tempt \
--batchSize 30 --testBatchSize 30 --mag_in 20 --foolmodel resnet18-imagenette --mode train \
--perturbation_type fix_trigger --target -1 --gpu_ids 0 --nEpochs 15