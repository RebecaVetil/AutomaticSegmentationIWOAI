from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import OWAIDataset
import os
import SegNet
import math
import datetime
from Layers import constant_initializer
import sys
import warnings

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # cpu 0
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

"""
Commands to run:

from personal computer:
python train.py --seg_type 'UNET' --data_dim '2D' --slices '0,160' --version 'both' --data_directory '/Volumes/MGAPRES/IWOAI/data' --epochs 1 --log_direct '/Volumes/MGAPRES/IWOAI/storing/NAME/log' --checkpoint_dir '/Volumes/MGAPRES/IWOAI/storing/NAME/ckpt'--model_dir '/Volumes/MGAPRES/IWOAI/storing/NAME/model'

from imperial computer:
python train.py --seg_type 'UNET' --data_dim '2D' --slices '0,160' --version 'both' --data_directory '/home/rbk/Desktop/IWOAIdata' --epochs 1 --log_direct '/home/rbk/Documents/storing/NAME/log' --checkpoint_dir '/home/rbk/Documents/storing/NAME/ckpt' --model_dir '/home/rbk/Documents/storing/NAME/model'

from pompeii:
python train.py --data_directory ./../data --patch_size 384 --log_direct ./../trainings/log --checkpoint_dir ./../trainings/ckpt --model_dir ./../trainings/model

"""
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
tf.app.flags.DEFINE_bool('transformation', True,
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


def placeholder_inputs(input_batch_shape, output_batch_shape):
    """
    This function generates placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded checkpoint file in the .run() loop, below.
    ---
    :param: input_batch_shape: the shape of the input batch
    :param: output_batch_shape: the shape of the output batch
    ---- 
    :return: images_placeholder: images placeholder.
    :return: segs_placeholder: segmentations placeholder.
    """

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    segs_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="segs_placeholder")   
   
    return images_placeholder, segs_placeholder

def soft_dice_coe(y_pred, y_true, loss_type='jaccard', spatial_axis = [1,2,3], smooth=1e-5):
    """
    This function computes the soft dice coefficient between two tensors
    ---
    :param: y_pred: predicted label, tensor of floats with shape [batch_size,..., num_channels]
    :param: y_true: true label, binary tensor of shape [batch_size,..., num_channels]
    :param: loss_type: string indicating the type of soft dice coefficient (Sorensen or Jaccard) 
    :param: spatial_axis: tuple indicating the spatial dimensions of y_pred, y_true
    :param: smooth : a float that will be added to the numerator and denominator.
    - If both output and target are empty, it makes sure dice is 1.
    - If either output or target are empty (all pixels are background), 
    then if smooth is very small, dice close to 0 (even the image values lower than the threshold).
    ---- 
    :return: the corresponding soft dice coefficient, a float between 0 and 1 (1 = totally match)
    """


    # calculate the commom elements, pixel by pixel
    no_common_element = tf.multiply(y_pred, tf.cast(y_true, y_pred.dtype))      # N HWD C
    # get the sum by class: reduce along HWD
    no_common_element = tf.reduce_sum(no_common_element, axis = spatial_axis)   # N C
    no_common_element = tf.cast(no_common_element, dtype = tf.float32)          

    # number of elements in each class
    no_ypred = tf.reduce_sum(y_pred, axis = spatial_axis)                       # N C
    no_ypred = tf.cast(no_ypred, dtype=tf.float32)
    no_ytrue = tf.reduce_sum(y_pred, axis = spatial_axis)                       # N C
    no_ytrue = tf.cast(no_ytrue, dtype=tf.float32)

    if loss_type == 'jaccard':
        num = no_common_element + tf.constant(smooth)                           # N C 
        denom = no_ytrue + no_ypred - no_common_element + tf.constant(smooth)   # N C

    elif loss_type == 'sorensen':
        num = tf.constant(2.0)* no_common_element+ tf.constant(smooth)          # N C
        denom = no_ytrue + no_ypred + tf.constant(smooth)                       # N C
    else:
        raise Exception("Unknown loss_type")

    ## Compute the coefficient
    coeff = num / denom                                                         # N C
    # Average over the batch and the channels
    dice = tf.reduce_mean(coeff)                                                # Scalar 

    return dice

def reverse_one_hot(one_hot_tensor, axis_channel, name= "reverse_encoding"):
    """
    This function reverses the one-hot-encoding of one_hot_tensor
    ---
    :param: one_hot_tensor: a binary tensor of shape [batch_size, spatial_dim, num_classes]
    :param: axis_channel: dimension along which perform the reverse one hot encoding
    ---- 
    :return: a tensor of shape [batch_size, spatial_dim] filled with integers from 0 to 6  (int 1,2,3,4,5,6 for biological masks, int 0 for background)
    """
    reverse = tf.argmax(one_hot_tensor, axis = axis_channel, name = name)
    return reverse 

def train():
    """Train the V/U-net model"""
    fig_captions = ['Background', 'Femoral_Cart', 'MedialTibial_Cart', 'LateralTibial_Cart', 'Patellar_Cart', 'Lateral_Meniscus', 'Medial_Meniscus']
    with tf.Graph().as_default():
        # Keep track of the number of batches seen so far
        global_step = tf.train.get_or_create_global_step() 

        # Sanity checks
        begin_slices = int(FLAGS.slices[0])
        end_slices = int(FLAGS.slices[1])
        if (end_slices - begin_slices < 0):
            sys.exit("Invalid slicing parameters");
        if (FLAGS.seg_type == 'UNET' and FLAGS.data_dim =='3D'):
            sys.exit("Invalid parameters: UNET can only support 2D data");
        
        assert isinstance(FLAGS.case_range, list)

        if ((FLAGS.case_select == 'all') and (FLAGS.case_range != ['1','61'])):
            sys.exit("Invalid case selection");
        elif ((FLAGS.case_select == 'random') and (int(FLAGS.case_range[0]) != int(FLAGS.case_range[1]))):
            print(FLAGS.case_select)
            print(FLAGS.case_range)
            sys.exit("Invalid random case selection");
        elif (FLAGS.case_select == 'select') :
            if (int(FLAGS.case_range[0]) >= int(FLAGS.case_range[1])):
                sys.exit("Invalid case selection");
            if ((1 <= int(FLAGS.case_range[0]) <= 59) and (2 <= int(FLAGS.case_range[1]) <= 61)):
                pass

        # Generate the shapes
        if (FLAGS.data_dim == '2D'):
            input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.input_channels)  # N HW C
            output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.num_classes) # N HW C
            channel_axis = 3 # channel axis 
        elif (FLAGS.data_dim == '3D'):
            input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, int(end_slices - begin_slices), FLAGS.input_channels)  # N HWD C
            output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, int(end_slices - begin_slices), FLAGS.num_classes) # N HWD C
            channel_axis = 4 # channel axis 
        else:
            sys.exit("Invalid VNET type (choose between 2D or 3D)");

        # Generate placeholder variables to represent the input tensors.
        images_placeholder, segs_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape) 
        
        for batch in range(FLAGS.batch_size):
            #images_log = images_placeholder[batch:batch+1,:,:,0,:] # convert types
            # Outputs a summary protocol buffer with images.
            #tf.summary.image("1image_SLICE0", images_log)
            for channel in range(0,FLAGS.num_classes):  
                if (FLAGS.data_dim == '2D'):
                    segs_log = tf.cast(tf.scalar_mul(255,segs_placeholder[batch:batch+1,:,:,channel:channel+1]), dtype=tf.uint8)
                    name_seg = '2seg_' + fig_captions[channel] +'_channel' + str(channel)
                elif (FLAGS.data_dim == '3D'):
                    segs_log = tf.cast(tf.scalar_mul(255,segs_placeholder[batch:batch+1,:,:,0,channel:channel+1]), dtype=tf.uint8)
                    name_seg = '2seg_SLICE0_' + fig_captions[channel] +'_channel' + str(channel)
                tf.summary.image(name_seg, segs_log)
            
        # Get images and segmentations
        train_data_dir = os.path.join(FLAGS.data_directory,'train')
        valid_data_dir = os.path.join(FLAGS.data_directory,'valid')

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):

            # create transformations to image and segmentations 
            # trainTransforms = []
            # validTransforms = []


            # create the train dataset
            TrainDataset = OWAIDataset.OWAIDataset(
                dataset_dim = FLAGS.data_dim,
                begin_slices = begin_slices,
                end_slices = end_slices,
                directory=train_data_dir,
                img_filename=FLAGS.image_filename,
                seg_filename=FLAGS.seg_filename,
                transforms=FLAGS.transformation,
                train=True,
                num_classes = FLAGS.num_classes,
                version = FLAGS.version,
                selection = FLAGS.case_select,
                selection_range = FLAGS.case_range,
                patch_size = FLAGS.patch_size
                )
            trainDataset = TrainDataset.create_dataset()
            #print('The training dataset has {} training examples obtained from {} versions'.format(trainDataset.nb_samples, FLAGS.version))
            trainDataset = trainDataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
            trainDataset = trainDataset.batch(FLAGS.batch_size)


            # create the valid dataset            
            ValidDataset = OWAIDataset.OWAIDataset(
                dataset_dim = FLAGS.data_dim,
                begin_slices = begin_slices,
                end_slices = end_slices,
                directory=valid_data_dir,
                img_filename=FLAGS.image_filename,
                seg_filename=FLAGS.seg_filename,
                transforms=FLAGS.transformation,
                train=False,
                num_classes = FLAGS.num_classes,
                version = FLAGS.version,
                selection = 'all',
                selection_range = (1,15),
                patch_size = FLAGS.patch_size
            )
            validDataset = ValidDataset.create_dataset()
            # print('The valid dataset has {} training examples obtained from {} versions'.format(validDataset.nb_samples,FLAGS.version))
            validDataset = validDataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
            validDataset = validDataset.batch(FLAGS.batch_size)


        # create the iterators  
        
        # Let's examine some of these augmented images
        # data_aug_iter = temp_ds.make_one_shot_iterator()
        # next_element = data_aug_iter.get_next()
        
        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        valid_iterator = validDataset.make_initializable_iterator()
        next_element_valid = valid_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            if (FLAGS.seg_type == 'UNET'):
                model = SegNet.UNet(
                    num_classes=FLAGS.num_classes, 
                    keep_prob=1.0, # default 1
                    num_channels=16, # default 16 
                    num_levels=4,  # default 4
                    num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
                    bottom_convolutions=3, # default 3
                    activation_fn="prelu") # default parametric relu
            elif (FLAGS.seg_type == 'VNET'):
                model = SegNet.VNet(
                    num_classes=FLAGS.num_classes, 
                    keep_prob=1.0, # default 1
                    num_channels=16, # default 16 
                    num_levels=4,  # default 4
                    num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
                    bottom_convolutions=3, # default 3
                    activation_fn="prelu") # default parametric relu
            else:
                sys.exit("Invalid Segmentation Net type (choose between UNET or VNET)");
            
            # obtain the logits
            logits = model.network_fn(images_placeholder) # [N HWD num_classes]

        for batch in range(FLAGS.batch_size):
            for channel in range(0,model.num_classes): 
                if (FLAGS.data_dim == '2D'):
                    logits_log = logits[batch:batch+1,:,:,channel:channel+1]
                elif (FLAGS.data_dim == '3D'):
                    logits_log = logits[batch:batch+1,:,:,0,channel:channel+1]
                name_summary = '3logitsRAW_' + fig_captions[channel] +'_channel' + str(channel)
                tf.summary.image(name_summary, logits_log)


        # # Exponential decay learning rate
        # train_batches_per_epoch = math.ceil(TrainDataset.data_size/FLAGS.batch_size)
        # decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        with tf.name_scope("learning_rate"):
            learning_rate = FLAGS.init_learning_rate
            #learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,global_step, decay_steps,FLAGS.decay_factor,staircase=True)
        #log_learning_rate = tf.summary.scalar('learning_rate', learning_rate)

        # Softmax operations on logits
        with tf.name_scope("softmax"):
            softmax_pred = tf.nn.softmax(logits,name="softmax", axis = -1) # same shape as logits, i.e., [N HWD C]
        
        for batch in range(FLAGS.batch_size):
            for channel in range(0,model.num_classes): 
                if (FLAGS.data_dim == '2D'):
                    softmax_log = softmax_pred[batch:batch+1,:,:,channel:channel+1]
                elif (FLAGS.data_dim == '3D'):
                    softmax_log = softmax_pred[batch:batch+1,:,:,0,channel:channel+1]
                name_softmax = '6softmaxRAW_' + fig_captions[channel] +'_channel' + str(channel)
                tf.summary.image(name_softmax, softmax_log)

        # Softmax Cross Entropy operations
        with tf.name_scope("softmax_cross_entropy"):
            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=reverse_one_hot(segs_placeholder, channel_axis)))
        log_cross_entropy = tf.summary.scalar('cross_entropy',xent)

        # Weighted Cross Entropy operations
        with tf.name_scope("weighted_cross_entropy"):
            class_weights = tf.constant([FLAGS.background_weight,1,1,1,1,1,1], dtype = tf.float32) # 7 classes
            # deduce weights for batch samples based on their true segmentation
            #onehot_segs = tf.one_hot(tf.squeeze(segs_placeholder,axis=[4]),depth = 2) #segs_placeholder= [N HWD num_classes]
            weights = tf.reduce_sum(class_weights * tf.cast(segs_placeholder, dtype=tf.float32), axis=-1)  #segs_placeholder= [N HWD num_classes] // reduce sum =[N HWD]
            # compute your (unweighted) softmax cross entropy loss
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=reverse_one_hot(segs_placeholder, channel_axis)) # logits [N HWD num_classes]
            # apply the weights, relying on broadcasting of the multiplication
            weighted_loss = unweighted_loss * tf.cast(weights,dtype =tf.float32)
            # reduce the result to get your final loss
            weighted_loss_op = tf.reduce_mean(weighted_loss)
        log_weighted_cross_entropy = tf.summary.scalar('weighted_xent',weighted_loss_op)  

        # Focal Loss operations
        with tf.name_scope("focal_loss"):
            gamma = FLAGS.gamma
            focal_loss=-tf.cast(segs_placeholder, tf.float32)*((1-softmax_pred)**gamma)*tf.log(softmax_pred+0.000000001)
            focal_loss = tf.reduce_sum(focal_loss,axis=-1) # [N HWD]
            focal_loss = tf.reduce_mean(focal_loss)
        log_focal_loss = tf.summary.scalar("focal_loss",focal_loss)  
        
        # Argmax Op to generate segmentation from softmax_pred
        with tf.name_scope("predicted_seg"):
            # reverse encode softmax_pred to get predictions: 
            pred = reverse_one_hot(softmax_pred, channel_axis, name="prediction") # [N HWD] 

        for batch in range(FLAGS.batch_size):
            if (FLAGS.data_dim == '2D'):
                pred_log = tf.cast(tf.scalar_mul(255,pred[batch:batch+1,:,:]), dtype=tf.uint8) # [1 HW]
                pred_log = tf.expand_dims(pred_log,axis=-1)
            elif (FLAGS.data_dim == '3D'):
                pred_log = tf.cast(tf.scalar_mul(255,pred[batch:batch+1,:,:,0:1]), dtype=tf.uint8) # [1 HWD]
            tf.summary.image("7predCASTMUL", pred_log)
            
            for channel in range(0,model.num_classes): 
                pred_channel_log = tf.where(tf.equal(pred_log,channel), tf.ones(tf.shape(pred_log), tf.uint8), tf.zeros(tf.shape(pred_log), tf.uint8))
                pred_channel_log = tf.cast(tf.scalar_mul(255,pred_channel_log), dtype = tf.uint8)
                name_pred = '8pred_' + fig_captions[channel] +'_channel' + str(channel)
                tf.summary.image(name_pred, pred_channel_log)

        # Accuracy of model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(pred, reverse_one_hot(segs_placeholder, channel_axis))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        log_accuracy = tf.summary.scalar('accuracy', accuracy)

        # Dice Similarity
        with tf.name_scope("dice"):
            sorensen = soft_dice_coe(softmax_pred,segs_placeholder, loss_type='sorensen', spatial_axis=[1,2,3])
            jaccard = soft_dice_coe(softmax_pred,segs_placeholder, loss_type='jaccard', spatial_axis=[1,2,3])
            sorensen_loss = 1. - sorensen
            jaccard_loss = 1. - jaccard
        log_sorensen = tf.summary.scalar('sorensen', sorensen)
        log_jaccard = tf.summary.scalar('jaccard', jaccard)
        log_sorensen_loss = tf.summary.scalar('sorensen_loss', sorensen_loss)
        log_jaccard_loss = tf.summary.scalar('jaccard_loss',jaccard_loss)

        # Training Op
        with tf.name_scope("training"):
            # optimizer
            if FLAGS.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == "nesterov_momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
            else:
                sys.exit("Invalid optimizer");

            # loss function
            if (FLAGS.loss_function == "xent"):
                loss_fn = xent
            elif(FLAGS.loss_function == "weight_xent"):
                loss_fn = weighted_loss_op
            elif(FLAGS.loss_function == "sorensen"):
                loss_fn = sorensen_loss
            elif(FLAGS.loss_function == "jaccard"):
                loss_fn = jaccard_loss
            elif(FLAGS.loss_function == "focal"):
                loss_fn = focal_loss
            else:
                sys.exit("Invalid loss function");

            train_op = optimizer.minimize(
                loss = loss_fn,
                global_step=global_step)

        # # epoch checkpoint manipulation
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        start_epoch_inc = start_epoch.assign(start_epoch+1)

        # saver
        summary_op_all = tf.summary.merge_all()
        summary_op_scalar = tf.summary.merge([log_sorensen, log_jaccard, log_sorensen_loss, log_jaccard_loss, log_accuracy, log_cross_entropy, log_weighted_cross_entropy, log_focal_loss])

        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir ,"checkpoint")
        print("Setting up Saver...")
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

        config = tf.ConfigProto(allow_soft_placement = True) #here
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        # training cycle
        with tf.Session(config=config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            print("{}: Start training...".format(datetime.datetime.now()))

            # summary writer for tensorboard
            train_summary_writer = tf.summary.FileWriter(FLAGS.log_direct + '/train', sess.graph)
            valid_summary_writer = tf.summary.FileWriter(FLAGS.log_direct + '/valid', sess.graph)

            # restore from checkpoint
            if FLAGS.restore_training:
                # check if checkpoint exists
                if os.path.exists(checkpoint_prefix+"-latest"):
                    print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
                    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest")
                    saver.restore(sess, latest_checkpoint_path)
            
            print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))
            # loop over epochs
            for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
                # initialize iterator in each new epoch
                sess.run(train_iterator.initializer)
                sess.run(valid_iterator.initializer)
                print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

                # training phase
                while True: # infinite loop
                    try:
                        [image, seg] = sess.run(next_element_train)
                        #image = image[:,:,:,:,np.newaxis]
                        #seg = seg[:,:,:,:,np.newaxis]
                        model.is_training = True;
                        if (tf.train.global_step(sess, global_step) % 5 == 0): # write all the summaries
                            train, summary = sess.run([train_op, summary_op_all], feed_dict={images_placeholder: image, segs_placeholder: seg})
                            train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                        else: # only write scalar summaries
                            train, summary = sess.run([train_op, summary_op_scalar], feed_dict={images_placeholder: image, segs_placeholder: seg})
                            train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                        print('global_step: %s' % tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError: # we looped over all the batch 
                        start_epoch_inc.op.run() # increment the epoch 
                        # print(start_epoch.eval())
                        # save the model at end of each epoch training
                        print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
                        if not (os.path.exists(FLAGS.checkpoint_dir)):
                            os.makedirs(FLAGS.checkpoint_dir,exist_ok=True)
                        saver.save(sess, checkpoint_prefix, 
                            global_step=tf.train.global_step(sess, global_step),
                            latest_filename="checkpoint-latest")
                        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
                        break
                
                # testing phase
                print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
                while True:
                    try:
                        [image, seg] = sess.run(next_element_valid)

                        #image = image[:,:,:,:,np.newaxis]
                        #seg = seg[:,:,:,:,np.newaxis]
                        
                        model.is_training = False;
                        loss, summary = sess.run([xent, summary_op_all], feed_dict={images_placeholder: image, segs_placeholder: seg})
                        valid_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        break

        # close tensorboard summary writer
        train_summary_writer.close()
        valid_summary_writer.close()

def main(argv=None):
    if not FLAGS.restore_training:
        # clear log directory
        if tf.gfile.Exists(FLAGS.log_direct):
            tf.gfile.DeleteRecursively(FLAGS.log_direct)
        tf.gfile.MakeDirs(FLAGS.log_direct)

        # clear checkpoint directory
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

        # clear model directory
        if tf.gfile.Exists(FLAGS.model_dir):
             tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)

    train()

if __name__=='__main__':
    tf.app.run()
