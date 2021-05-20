# CUDA_VISIBLE_DEVICES=0,1 python generator.py \
# --expname test1 \
# --batchSize 30 --testBatchSize 1 --mag_in 15 --foolmodel resnet18-imagenette --mode train \
# --perturbation_type universal --target -1 --gpu_ids 0,1 --nEpochs 10

CUDA_VISIBLE_DEVICES=0,1 python generator.py \
--expname testCifar \
--batchSize 30 --testBatchSize 1 --mag_in 5 --foolmodel vgg16-cifar10 --mode train \
--perturbation_type universal --target -1 --gpu_ids 0,1 --nEpochs 10