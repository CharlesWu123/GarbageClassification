python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
  --cfg configs/garbage/swinv2_base_patch4_window8_256.yaml \
  --accumulation-steps 2 \
  --eval \
  --resume /mnt/trained_model/swinv2_base_patch4_window8_256/202211181548/ckpt_epoch_best.pth
