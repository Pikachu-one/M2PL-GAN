
## M2PL-GAN: Multi-View Multi-Level Pathology Semantic Perception Learning for H\&E-to-IHC Virtual Staining


## Train Code

```bash
# PR
python train.py --gpu_ids 0 --dataroot /MIST/PR-001/PR/TrainValAB --name train --checkpoints_dir ./checkpoints/PR --model m2plgan --CUT_mode FastCUT --n_epochs 70 --n_epochs_decay 10 --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=64

# HER2
python train.py --gpu_ids 0 --dataroot /MIST/HER2-003/HER2/TrainValAB --name train  --checkpoints_dir ./checkpoints/HER2 --model m2plgan --CUT_mode FastCUT --n_epochs 70 --n_epochs_decay 10 --netD n_layers --ndf 32  --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5  --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=64

# KI67
python train.py --gpu_ids 0 --dataroot /MIST/KI67-001/PR/TrainValAB --name train --checkpoints_dir checkpoints/KI67 --model m2plgan --CUT_mode FastCUT --n_epochs 70 --n_epochs_decay 10 --netD n_layers --ndf 32  --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance  --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=64

# ER
python train.py --gpu_ids 0 --dataroot /MIST/ER-004/ER/TrainValAB --name train  --checkpoints_dir checkpoints/ER --model m2plgan --CUT_mode FastCUT --n_epochs 70 --n_epochs_decay 10 --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=64

# BCI
python train.py --gpu_ids 0 --dataroot /BCI/BCI_dataset --name train  --checkpoints_dir ./checkpoints/BCI --model m2plgan --CUT_mode FastCUT  --n_epochs 40 --n_epochs_decay 10 --netD n_layers --ndf 32  --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance  --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0  --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1  --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=64

# NEFU-JF
python train.py --gpu_ids 0 --dataroot /NEFU-JF/TrainValAB --name train  --checkpoints_dir ./checkpoints/NEFU-JF --model m2plgan --CUT_mode FastCUT  --n_epochs 40 --n_epochs_decay 10 --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance  --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 1.0  --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1  --load_size 1024 --crop_size 512 --preprocess crop --flip_equivariance False  --display_winsize 512 --update_html_freq 100 --save_epoch_freq 5 --CACM=1.0 --LDAM=10.0 --patchwpatch --GBCLM=0.1 --conv_type=tag --patchsize=128
```
## Generate Virtual Staining Images

```bash
#PR
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/MIST/PR-001/PR/TrainValAB --name train --checkpoints_dir ./checkpoints/PR --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val   

#HER2
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/MIST/HER2-003/HER2/TrainValAB --name train --checkpoints_dir ./checkpoints/HER2 --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val    

#KI67
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/MIST/KI67-002/KI67/TrainValAB --name train --checkpoints_dir ./checkpoints/KI67 --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val    

#ER
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/MIST/ER-004/ER/TrainValAB --name train --checkpoints_dir ./checkpoints/ER --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256--dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val    

#BCI
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/BCI/BCI_dataset --name train --checkpoints_dir ./checkpoints/BCI --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256   --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 1000 --phase val    

#NEFU-JF
#python test.py   --gpu_ids 0 --dataroot /mnt/a-1/liuzequn/GAN_DATA/NEFU-JF/TrainValAB --name train --checkpoints_dir ./checkpoints/NEFU-JF --model m2plgan --CUT_mode FastCUT --netD n_layers --ndf 32 --netG resnet_6blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256  --dataset_mode aligned --direction AtoB --num_threads 15 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess none --flip_equivariance False --display_winsize 512 --num_test 441 --phase val   
```

## Evaluate

**Run evaluation scripts:**

```bash
python evaluate.py
python scripts/eval_pathology_metric.py
python scripts/pearson_R.py
```

For R<sub>ave</sub>  detailed calculate, please refer to the paper  
**USIGAN: Unbalanced Self-Information Feature Transport for Weakly Paired Image IHC Virtual Staining**.

