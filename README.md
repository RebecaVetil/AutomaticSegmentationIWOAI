# AutomaticSegmentationIWOAI

## Presentation (not finished)
### Presentation of the repository
This repository gathers the code of my Master's Thesis in Biomedical Engineering, completed at Imperial College London from May to September 2019.

This code performs automatic segmentation of knee IRM. The data has been provided by the OAI, for the IWOAI 2019 Automatic segmentation Challenge.

I implemented a UNET and a VNET.

### Presentation of the challenge

## Description of the data
### Basic Description

We are provided with:
- a training dataset, made of 60 patients
- a validation dataset, made of 14 patients


- The shape of an image scan is: (384, 384, 160)
- The type of an image scan is: float32
- The shape of a segmentation scan is: (384, 384, 160, 6)
- - The 4-th dimension of the segmentation scans corresponds to the 6 masks we have to produce: 'Femoral Cart', 'Medial Tibial Cart', 'Lateral Tibial Cart', 'Patellar Cart', 'Lateral Meniscus', and 'Medial Meniscus'.
- The type of a segmentation scan is: uint8


_____________________
TODO 
### Challenges

## Learning Strategy
- Always: crop, normalize
- Choose between GPU, CPU, UNET, 2D VNET, 3D VNET
>> If 3D, reverse the versions
>> If 2D, try to compare UNET, VNET 2D, UNET with skip connections
- Proceed by stages:
- - 50 % of the data, emphasize : learning background
- - 25 %
- - 25 % 
- - 100 % with V2 - fine tuning 
- Test different losses, optimizer 
- Hyperparameters

## Parameters:

-- seg_type UNET --data_dim 2D --slices '0,160' --version both --case_select all --case_range '1,61' --data_directory /Volumes/MGAPRES/IWOAI/data --image_filename img.npy --input_channels 1 --sef_filename seg.npy --num_classes 7 --batch_size 1 --patch_size 300 --transformation False --epochs 1 --log_direct '/Volumes/MGAPRES/IWOAI/tmp_wxent01/log' --init_learning_rate 1e-2 --decay_factor 0.01 --decay_steps 100 --display_step 10 --save_interval 1 --checkpoint_dir '/Volumes/MGAPRES/IWOAI/tmp_wxent01/ckpt' --model_dir /Volumes/MGAPRES/IWOAI/tmp_wxent01/model --restore_training False --drop_ratio 0 --min_pixel 500 --shuffle_buffer_size 1 --loss_function weight_xent --background_weight 0.1 --gamma 2 --optimizer seg --momentum 0.5


--seg_type UNET VNET
--data_dim 2D 3D
--slices '0,160' 
--version both V00 V01
--case_select all random select 
--case_range '1,61' 'n,n' 'n,m'
--data_directory /Volumes/MGAPRES/IWOAI/data 
--image_filename img.npy 
--input_channels 1 
--sef_filename seg.npy 
--num_classes 7 
--batch_size 1 
--patch_size 384
--transformation False True
--epochs 1 
--log_direct /Volumes/MGAPRES/IWOAI/tmp_wxent01/log
--init_learning_rate 1e-2 
--decay_factor 0.01 
--decay_steps 100 
--display_step 10 
--save_interval 1 
--checkpoint_dir /Volumes/MGAPRES/IWOAI/tmp_wxent01/ckpt
--model_dir /Volumes/MGAPRES/IWOAI/tmp_wxent01/model
--restore_training False True
--drop_ratio 0 
--min_pixel 500 
--shuffle_buffer_size 1 
--loss_function weight_xent, xent, sorensen, jaccard, focal
--background_weight 0.1 
--gamma 2 
--optimizer sgd adam momentum nesterov_momentum
--momentum 0.5


## Credits:

https://github.com/jackyko1991/vnet-tensorflow
https://github.com/MiguelMonteiro/VNet-Tensorflow
