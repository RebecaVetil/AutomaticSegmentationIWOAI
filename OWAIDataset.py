import tensorflow as tf
import os
import numpy as np
import math



def read_np(image_path, segm_path):
    """ Loads and returns the image and the segmentation """
    n_classes = 7
    img = np.load(image_path.decode())  #384 384 160 1
    seg = np.load(segm_path.decode()) #384 384 160 6

    # Adding the background as a mask
    seg_new = np.zeros(np.shape(seg)[:-1] + tuple([n_classes])) # shape (384, 384, 160, 7)
    seg_new[:,:,:,1:n_classes] = seg # biological masks at channels 1,2,3,4,5,6
    seg_new[...,0] = np.where(np.sum(seg, axis=3) ==0, 1, 0) # background at channel 0 

    return img, seg_new.astype('uint8') 

def slice_tf(image, segm, begin, end):
    image = image[:,:,begin:end,:]
    segm = segm[:,:,begin:end,:]
    return image, segm

def split_tf(image, segm,no_slices):
    """ Split the 3D data into 2D data """ 
    imgs = tf.split(image, no_slices, axis = 2, name = 'split_image')
    segs = tf.split(segm, no_slices, axis = 2, name = 'split_seg')
    imgs = tf.squeeze(imgs, axis = -1)
    segs = tf.squeeze(segs)
    return imgs, segs


class OWAIDataset(object):
    """
    Load image and label for training, testing, validation.
    """

    def __init__(self,
        dataset_dim,
        begin_slices = 0,
        end_slices = 160,
        directory = '/Volumes/MGAPRES/IWOAI/data/valid',
        img_filename = 'img.npy',
        seg_filename = 'seg.npy',
        transforms = None,
        train = False,
        num_classes = 7, 
        version = 'both'):
        """ Class Constructor """

        # Initialise the variables of the instance
        self.directory = directory
        self.img_filename = img_filename
        self.seg_filename = seg_filename
        self.transforms = transforms
        self.train = train
        self.num_classes = num_classes
        self.type = dataset_dim
        self.begin_slices = begin_slices 
        self.end_slices = end_slices
        self.version = version

    def create_dataset(self):
        """ Create the TensorFlow dataset associated with the OWAIDataset object"""
        img_paths = []
        seg_paths = []

        # creating the list of paths
        for case in os.listdir(self.directory): 
            # one case = one patient + one version
            # one image is stored as directory/trainORvalid/case/img.npy
            if ((case[-3:] == self.version) or (self.version =='both')):
                img_paths.append(os.path.join(self.directory,case,self.img_filename))
                # one segmentation is stored as directory/trainORvalid/case/seg.npy
                seg_paths.append(os.path.join(self.directory,case,self.seg_filename))
        
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,seg_paths)) 
        dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(read_np, [image_path, label_path], [tf.float32,tf.uint8])))
        no_slices = int(self.end_slices - self.begin_slices)
        if not (self.begin_slices ==0 and self.end_slices ==160):
            # we slice the tensor so as we get only the selected slices
            dataset = dataset.map(lambda img_3D, seg_3D : slice_tf(img_3D, seg_3D,self.begin_slices, self.end_slices))
        if (self.type == '2D'):
            # we unbatch the data in order to get a 2D dataset
            dataset = dataset.map(lambda img_3D, seg_3D : split_tf(img_3D, seg_3D, no_slices))
            dataset = dataset.apply(tf.contrib.data.unbatch())

        self.dataset = dataset
        self.nb_samples = len(img_paths)

        return self.dataset
