# 三个参数：显卡号、工作文件夹、实验名，如：./train.sh 0 Cy3D_SegMap_GS segmap_debug
CUDA_VISIBLE_DEVICES=$1 python trainer.py ./experiments/$2 $3
