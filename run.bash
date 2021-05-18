CUDA_VISIBLE_DEVICES=0,1 python generator.py \
--expname \test \
--batchSize 30 --testBatchSize 1 --mag_in 15 --foolmodel resnet18-imagenette --mode train \
--perturbation_type universal --target -1 --gpu_ids 0,1 --nEpochs 10