##########################
#  Credits: Mingle Xu	 #
##########################
# In order to run this script over a separate test directory please rename it under "val"
# or feel free to edit the code in the file engine_finetune.py to consider so.
export IMAGENET_DIR="/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/" # Data directory.
export name="baseline_run_1" # Checkpoints directory.
export epoch=100
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch main_finetune.py \
--eval \
--resume "checkpoint/${name}/checkpoint-${epoch}.pth" \
--model vit_large_patch16 \
--batch_size 1 \
--data_path ${IMAGENET_DIR} \
--nb_classes 80000 \
--visualize_epoch "${epoch}" \
--max_num 30 # fix this.