# should finish within seconds
python  train.py  --device cuda:0 --n_channels 32  --n_residual_blocks 32 --kernel_size1 5 --kernel_size2 5 --n_upscale_layers 1 --downsample_data 2 --downsample_label 1 --epochs 2 > run.out
