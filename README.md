# AutomaticSegmentationIWOAI

## Presentation (not finished)
### Presentation of the repository
This repository gathers the code of my Master's Thesis in Biomedical Engineering, completed at Imperial College London from May to September 2019.

This code performs automatic segmentation of knee IRM. The data has been provided by the OAI, for the IWOAI 2019 Automatic segmentation Challenge.

I implemented a UNET and a VNET.

### Presentation of the challenge

## Description of the data
### Basic Description

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


tf.app.flags.DEFINE_string('seg_type', 'UNET',
    """ 'UNET' or 'VNET'""")
tf.app.flags.DEFINE_string('data_dim', '2D',
    """If VNET, '2D' or '3D'""")
tf.app.flags.DEFINE_list("slices", '0,160',
    """Begining and end of slicing ('0,160' means selecting all the dataset)""") 
tf.app.flags.DEFINE_string("version", 'both',
    """ Which timepoint data to use: 'V00', 'V01' or 'both'""")
tf.app.flags.DEFINE_string("case_select", 'all',
    """ Cases to select for the training: all (60 patients), random (select n random patients) or select (select patients within a slice)'""")
tf.app.flags.DEFINE_list("case_range", '1,61',
    """Begining and end of slicing for the case selection: if all, must be (1,61) / if random, must be (n,n), n int / if select, must be (n,m), n and m int""")    
tf.app.flags.DEFINE_string('data_directory', '/Volumes/MGAPRES/IWOAI/data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_string('image_filename','img.npy',
    """Image filename""")
tf.app.flags.DEFINE_integer('input_channels',1,
    """Number of channels of the input images: 1 (Grayscale), 3 (RGB)""")
tf.app.flags.DEFINE_string('seg_filename','seg.npy',
    """Segmentation filename""")
tf.app.flags.DEFINE_integer('num_classes',7,
    """Number of classes of the segmentation""") #6 biological masks (channels 1,2,3,4,5,6) and a background mask (channel 0)
tf.app.flags.DEFINE_integer('batch_size',1,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',300,
    """Size (pixels) of a data patch. Here, 300 because we are cropping the image.""")
tf.app.flags.DEFINE_bool('transformation', False,
    """Wheter or not to apply the transformations""")
tf.app.flags.DEFINE_integer('epochs',1,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_direct', '/Volumes/MGAPRES/IWOAI/tmp_wxent01/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.01,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
    """Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',10,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1,
    """Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/Volumes/MGAPRES/IWOAI/tmp_wxent01/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_string('model_dir','/Volumes/MGAPRES/IWOAI/tmp_wxent01/model',
    """Directory to save model""")
tf.app.flags.DEFINE_bool('restore_training',False,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0,
    """Probability to drop a cropped area if the segmentation is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',500,
    """Minimum non-zero pixels in the cropped segmentation""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',1,
    """Number of elements used in shuffle buffer""")
tf.app.flags.DEFINE_string('loss_function','weight_xent',
    """Loss function used in optimization (xent, weight_xent, sorensen, jaccard, focal)""")
tf.app.flags.DEFINE_float('background_weight',0.1,
    """Background weight when computing the weighted cross entropy""")    
tf.app.flags.DEFINE_integer('gamma',2,
    """Gamma parameter used for the focal loss""")
tf.app.flags.DEFINE_string('optimizer','sgd',
    """Optimization method (sgd, adam, momentum, nesterov_momentum)""")
tf.app.flags.DEFINE_float('momentum',0.5,
    """Momentum used in optimization""")

## Credits:

https://github.com/jackyko1991/vnet-tensorflow
https://github.com/MiguelMonteiro/VNet-Tensorflow