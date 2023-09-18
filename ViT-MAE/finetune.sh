#!/bin/bash

##############################################################################################
# Resume training from a given epoch if the program stop occasionally (uncomment lines below)#
##############################################################################################

#export IMAGENET_DIR='/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/'
#export name="IN1k_Clef2022"
#export all_epoch=100 # Total No. of epochs
#export resume_epoch=45 # The epoch no. to resume the training process
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size 128 \
#    --model vit_large_patch16  \
#    --epochs ${all_epoch} \
#    --blr 7.5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir "checkpoint/${name}/log" \
#    --nb_classes 80000 \
#    --resume "./checkpoint/${name}/checkpoint-${resume_epoch}.pth" --start_epoch ${resume_epoch} \
#    --eval_epoch 1 \
#    --save_model_epoch 2 \
#    --output_dir checkpoint/${name}

###############################
# Start training from epoch 0 #
###############################
export IMAGENET_DIR='LifeCLEFPlant2022/web' # The data's root directory (containing train & val directories)
export PRETRAIN_CHKPT='LifeCLEFPlant2022/ViT/mae/mae_pretrain_vit_large.pth' # Path for the pre-trained checkpoint
export name="PC2022-BASELINE-WEB-ONLY" # Name of the directory in which the checkpoints will be saved (i.e., 'checkpoint/{name}/checkpoint_xx.pth')
export all_epoch=101 # Number of epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 main_finetune.py \
    --accum_iter 4 \
    --batch_size 192 \
    --model vit_large_patch16  \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${all_epoch} \
    --blr 4.5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --log_dir "checkpoint/${name}/log" \
    --nb_classes 57309 \
    --eval_epoch 1 \
    --save_model_epoch 5 \
    --output_dir checkpoint/${name} \
    --num_workers 20

