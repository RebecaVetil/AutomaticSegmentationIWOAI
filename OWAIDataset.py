import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random 
from scipy import ndimage

def read_np(image_path, segm_path, n_classes):
    """ Loads and returns the image and the segmentation """
    img = np.load(image_path.decode())  #384 384 160 1
    seg = np.load(segm_path.decode()) #384 384 160 6

    # Adding the background as a mask
    seg_new = np.zeros(np.shape(seg)[:-1] + tuple([n_classes])) # shape (384, 384, 160, 7)
    seg_new[:,:,:,1:n_classes] = seg # biological masks at channels 1,2,3,4,5,6
    seg_new[...,0] = np.where(np.sum(seg, axis=3) ==0, 1, 0) # background at channel 0
    return img, seg_new.astype('uint8') 

def transformations(img, seg, size):
    ###########LIST OF TRANSFORMATIONS###########
    transforms = [RandomCrop(output_size= size.item()),ManualNormalization(0, 0.005), ]
    #############################################

    depth_dim = seg.shape[3]

    # Going into sitk formats to do the transformations
    img_t = np.transpose(img,(2,1,0,3)) 
    seg_t = np.transpose(seg,(2,1,0,3))
    
    img_trans = sitk.GetImageFromArray(np.squeeze(img_t), isVector=False) #384 384 160

    seg_trans = []
    for t in range(depth_dim):
        seg_trans.append(sitk.GetImageFromArray(np.asarray(seg_t[...,t], np.uint8), False)) #a list of depth_dim (384, 384, 160) images
        
    sample = {'image':img_trans, 'label': seg_trans} # current sample with the image and the label

    # if transformations are specified, apply successively each transformation to the sample
    for transform in transforms:
        if transform.name =='Confidence Crop':
            # this transformation requires to work on the reverse one hot encoding
            sample = transform(sample, seg_t)
        else:
            sample = transform(sample) 
    
    # Returning to the numpy format
    img_trans = sitk.GetArrayFromImage(sample['image'])
    img_trans = img_trans[...,np.newaxis]
    
    seg_trans = sitk.JoinSeries(sample['label'])
    seg_trans = sitk.GetArrayFromImage(seg_trans)
    seg_trans = np.asarray(seg_trans,np.uint8)

    img_trans = np.transpose(img_trans,(2,1,0,3)) 
    seg_trans = np.transpose(seg_trans,(3,2,1,0))

    return img_trans, seg_trans
    
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

def downsample(img, seg, times):
    ## DOWNSAMPLING
    kernel_img = np.array([[0.25,0.25],[0.25,0.25]])
    kernel_img = kernel_img[:,:,np.newaxis,np.newaxis]

    kernel_seg = np.array([[0.5,0.5],[0.5,0.5]])
    kernel_seg = kernel_seg[:,:,np.newaxis,np.newaxis]

    img_downsampled = img
    seg_downsampled = seg

    for k in range(0,times): 
        img_downsampled = ndimage.convolve(img_downsampled, kernel_img)[:img_downsampled.shape[0]:2,:img_downsampled.shape[1]:2,:]
        seg_downsampled = ndimage.convolve(seg_downsampled, kernel_seg)[:seg_downsampled.shape[0]:2,:seg_downsampled.shape[1]:2,:]


    return img_downsampled, seg_downsampled

def downsample_tf(img, seg):
    # shapes : img  H W D 1 and seg  H W D 7
    # to use tf.image.resize_images function, 
    # we need to work on these shapes.
    new_size = 75
    num_classes = 7
    # Pre-pocessing
    img = tf.expand_dims(img, axis = 0, name = "add_batch_size_img") # 1 H W D 1
    img = tf.squeeze(img, axis = -1, name = "squeeze_chanels_img") # 1 H W D
    seg = tf.argmax(seg, axis = 3, name = 'squeeze_channels_seg') # H W D
    seg = tf.expand_dims(seg, axis = 0,name= "add_batch_size_seg") # 1 H W D

    # Method 3 (Area) for Images
    img_downsampled = tf.image.resize_images(img, [new_size,new_size], method = 3)
    # Method 1 (Nearest Neighbours) for Seg
    seg_downsampled = tf.image.resize_images(seg, [new_size,new_size], method = 1)

    # Post-processing
    img_downsampled = tf.squeeze(img_downsampled, axis = 0, name = "squeeze_batch_size_img") # H W D
    img_downsampled = tf.expand_dims(img_downsampled, axis = -1, name = "add_chanels_img") # H W D 1

    seg_downsampled = tf.squeeze(seg_downsampled, axis = 0, name = "squeeze_batch_size_seg") # H W D
    seg_downsampled = tf.one_hot(seg_downsampled, depth = num_classes, axis = -1, name = "add_chanels_seg") # H W D 7

    return img_downsampled, seg_downsampled

def upsample(img, seg, times):
    factor = pow(2,times)

    ## UPSAMPLING
    img_upsampled = ndimage.zoom(img, [factor,factor,1,1], order=1)
    seg_upsampled = ndimage.zoom(seg, [factor,factor,1,1], order=0)
    
    return img_upsampled, seg_upsampled

def upsample_tf(img, seg, original_size, num_classes):
    # Pre-pocessing
    img = tf.expand_dims(img, axis = 0, name = "add_batch_size_img") # 1 H W D 1
    img = tf.squeeze(img, axis = -1, name = "squeeze_chanels_img") # 1 H W D

    seg = tf.argmax(seg, axis = 3, name = 'squeeze_channels_seg') # H W D
    seg = tf.expand_dims(seg, axis = 0,name= "add_batch_size_seg") # 1 H W D

    # Method 3 (Area) for Images
    img_upsampled = tf.image.resize_images(img, [original_size,original_size], method = 3,align_corners=False, name = "up_img_ar")
    # Method 1 (Nearest Neighbours) for Seg
    seg_upsampled = tf.image.resize_images(seg, [original_size,original_size], method = 1, align_corners=False, name = "up_seg_nn")
    
    # Post-processing
    img_upsampled = tf.squeeze(img_upsampled, axis = 0, name = "squeeze_batch_size_img") # H W D
    img_upsampled = tf.expand_dims(img_upsampled, axis = -1, name = "add_chanels_img") # H W D 1

    seg_upsampled = tf.squeeze(seg_upsampled, axis = 0, name = "squeeze_batch_size_seg") # H W D
    seg_upsampled = tf.one_hot(seg_upsampled, depth = num_classes, axis = -1, name = "add_chanels_seg") # H W D 7

    return img_upsampled, seg_upsampled

def patch(img, seg, size, num_slices, num_classes):
    # IMG
    img = tf.expand_dims(img, axis = 0, name = "add_batch_size")
    img = tf.squeeze(img, axis = -1, name = "squeeze")
    img_patched = tf.image.extract_image_patches(images = img,
                                            ksizes=[1]+ [size, size] +[1],
                                            strides=[1]+ [size/2, size/2] +[1],
                                            rates = [1, 1, 1, 1],
                                            padding = "VALID",
                                            name = "patching_image")
    n = tf.shape(img_patched)[1]
    img_patched = tf.reshape(img_patched, [n*n,size,size, num_slices], name = "reshaping")
    img_patched = tf.expand_dims(img_patched, axis = -1, name = "add_channel_axis")

    # SEG
    seg = tf.argmax(seg, axis = 3, name = 'one_hot_encoding')
    seg = tf.expand_dims(seg, axis = 0,name= "expand_dim")
    seg_patched = tf.image.extract_image_patches(images = seg,
                                            ksizes=[1]+ [size, size] +[1],
                                            strides=[1]+ [size/2, size/2] +[1],
                                            rates = [1, 1, 1, 1],
                                            padding = "VALID",
                                            name = "patching_image")
    seg_patched = tf.reshape(seg_patched, [n*n,size,size,num_slices], name = "reshaping")
    seg_patched = tf.one_hot(seg_patched,depth = num_classes, axis = -1, name = "one_hot_encoding")
    
    return img_patched, seg_patched


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
        transforms = 'False',
        train = False,
        num_classes = 7, 
        version = 'both',
        selection ='all',
        selection_range = (1,61),
        image_size =384,
        patch_size = 100,
        down_factor = 2):
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
        self.selection = selection
        self.selection_range = selection_range
        self.image_size = image_size
        self.patch_size = patch_size
        self.down_factor = down_factor

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

        # case01_V00 ends up at the end of the lists
        # we manually put it at the first position
        img_paths.insert(0,img_paths[-1])
        del(img_paths[-1])

        seg_paths.insert(0,seg_paths[-1])
        del(seg_paths[-1])

        # Selecting only some cases
        if (self.selection == 'random'):
            idx = random.sample(range(len(img_paths)), int(self.selection_range[0]))
            img_paths = [img_paths[i] for i in idx]
            seg_paths = [seg_paths[i] for i in idx]
            
        elif (self.selection == 'select'):
            if (self.version =='both'):
                idx = [i for i in range(2*int(self.selection_range[0])-2, 2*int(self.selection_range[1])-2)]
            else :
                idx = [i for i in range(int(self.selection_range[0]), int(self.selection_range[1]))]
            img_paths = [img_paths[i] for i in idx]
            seg_paths = [seg_paths[i] for i in idx]

        # if self.selection == 'all', we keep all the cases

        dataset = tf.data.Dataset.from_tensor_slices((img_paths,seg_paths)) 
        dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(read_np, [image_path, label_path, self.num_classes], [tf.float32,tf.uint8])))
        if (self.transforms == 'True'):
            dataset = dataset.map(lambda img, label: tuple(tf.py_func(transformations, [img, label, self.image_size], [tf.float32,tf.uint8])))

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

    def downsample_dataset(self):
        new_size = int(self.image_size/pow(2,self.down_factor))
        num_classes = self.num_classes
        self.dataset = self.dataset.map(lambda img, label: tuple(tf.py_func(downsample, [img, label, self.down_factor], [tf.float32,tf.uint8])))
        #self.dataset = self.dataset.map(downsample_tf)
        return self.dataset
    
    def upsample_dataset(self):
        #self.dataset = self. dataset.map(lambda img, label: tuple(tf.py_func(upsample, [img, label, self.down_factor], [tf.float32,tf.uint8])))
        self.dataset = self.dataset.map(lambda img, label: upsample_tf(img, label, self.image_size, self.num_classes))
        return self.dataset
    
    def patch_dataset(self):
        size = self.patch_size
        num_slices = int(self.end_slices - self.begin_slices)
        num_classes = self.num_classes
        self.dataset = self.dataset.map(lambda img, label : patch(img, label, size, num_slices, num_classes))
        return self.dataset

#### TRANSFORMATIONS

class Normalization(object):
    """
    Normalize an image to 0 - 255
    """

    def __init__(self):
        self.name = 'Normalization'

    def __call__(self, sample):
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMaximum(0.005)
        rescaleFilter.SetOutputMinimum(0)
        image, label = sample['image'], sample['label']
        image = rescaleFilter.Execute(image)

        return {'image': image, 'label': label}

class StatisticalNormalization(object):
    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        statisticsFilter = sitk.StatisticsImageFilter()
        statisticsFilter.Execute(image)

        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(0.005)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.sigma*statisticsFilter.GetSigma())
        intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.sigma*statisticsFilter.GetSigma())

        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}

class ManualNormalization(object):
    """
    Normalize an image by mapping intensity with given max and min window level
    """

    def __init__(self,windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(self.windowMax)
        intensityWindowingFilter.SetWindowMinimum(self.windowMin)

        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}

class Invert(object):
    """
    Invert the image intensity from 0-255 
    """

    def __init__(self):
        self.name = 'Invert'

    def __call__(self, sample):
        invertFilter = sitk.InvertIntensityImageFilter()
        image = invertFilter.Execute(sample['image'],0.005)
        label = sample['label']

        return {'image': image, 'label': label}

class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
    Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
    This transformation only applicable in train mode
    Args:
    output_size (tuple or int): Desired output size. If int, quadratic crop is made.
    """

    def __init__(self, output_size=(300,300), drop_ratio=0.3, min_pixel=1, original_depth_size = 160):

        self.name = 'Random Crop'
        
        assert isinstance(original_depth_size,int)
        
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, original_depth_size)
        else: 
            assert isinstance(output_size, tuple)
            if len(output_size) == 2:
                self.output_size = output_size + tuple([original_depth_size])
            else:
                assert len(output_size) == 3
                self.output_size = output_size

        assert isinstance(drop_ratio, (int,float))
        if drop_ratio >=0 and drop_ratio<=1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')

        assert isinstance(min_pixel, int)
        if min_pixel >=0 :
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')

    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        depth_dim = len(label)
        size_old = image.GetSize()
        size_new = self.output_size
        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0],size_new[1], size_new[2]])

        while not contain_label: 
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0]-size_new[0])

            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1]-size_new[1])
                
            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2]-size_new[2])

            roiFilter.SetIndex([start_i,start_j, start_k])

            label_crop=[]
            empty_label = []
            for i in range(depth_dim):
                label_crop.append(roiFilter.Execute(label[i]))
                statFilter = sitk.StatisticsImageFilter()
                statFilter.Execute(label_crop[i])
                if statFilter.GetSum()<self.min_pixel:
                    # empty label for this crop
                    empty_label.append(True)
            # will iterate until a sub volume containing at least 3 labels is extracted
            if (sum(empty_label) > (float(depth_dim - 1) / 2)):
                if (random.random() <= self.drop_ratio):
                    contain_label = True
            else:
                contain_label = True
                           
        image_crop = roiFilter.Execute(image)

        return {'image': image_crop, 'label': label_crop}

class RandomNoise(object):
    """
    Randomly noise to the image in a sample. This is usually used for data augmentation.
    """
    def __init__(self):
        self.name = 'Random Noise'

    def __call__(self, sample):
        self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
        self.noiseFilter.SetMean(0)
        self.noiseFilter.SetStandardDeviation(0.001) #0.0001: still good quality #0.001: significant noise, but ok for the eye #0.01: spoiled image

        image, label = sample['image'], sample['label']
        image = self.noiseFilter.Execute(image)

        return {'image': image, 'label': label}

class ConfidenceCrop(object):
    """
    Crop the image in a sample that is certain distance from individual labels center. 
    This is usually used for data augmentation with very small label volumes.
    The distance offset from connected label centroid is model by Gaussian distribution with mean zero and user input sigma (default to be 2.5)
    i.e. If n isolated labels are found, one of the label's centroid will be randomly selected, and the cropping zone will be offset by following scheme:
    s_i = np.random.normal(mu, sigma*crop_size/2), 1000)
    offset_i = random.choice(s_i)
    where i represents axis direction
    A higher sigma value will provide a higher offset

    Args:
    output_size (tuple or int): Desired output size. If int, cubic crop is made.
    sigma (float): Normalized standard deviation value.
    """

    def __init__(self, output_size=(300,300), sigma=2.5, original_depth_size = 160):
        self.name = 'Confidence Crop'

        assert isinstance(original_depth_size,int)
        assert isinstance(output_size, (int, tuple))
        assert isinstance(sigma, (float, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, original_depth_size)
        else: 
            assert isinstance(output_size, tuple)
            if len(output_size) == 2:
                self.output_size = output_size + tuple([original_depth_size])
            else:
                assert len(output_size) == 3
                self.output_size = output_size

        
        if isinstance(sigma, float) and sigma >= 0:
            self.sigma = (sigma,sigma)
        else:
            assert len(sigma) == 2
            self.sigma = sigma

    def __call__(self,sample, seg):
        # seg is a numpy array on which we perform the reverse one hot encoding
        
        seg = np.argmax(seg, axis = 3)
        seg = sitk.GetImageFromArray(seg, isVector=False)
            
        image, label = sample['image'], sample['label']
        depth_dim = len(label)
        
        size_new = self.output_size

        # guarantee label type to be integer
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkInt8)
        seg = castFilter.Execute(seg)

        ccFilter = sitk.ConnectedComponentImageFilter()
        segCC = ccFilter.Execute(seg)

        segShapeFilter = sitk.LabelShapeStatisticsImageFilter()
        segShapeFilter.Execute(segCC)

        if segShapeFilter.GetNumberOfLabels() == 0:
            # handle image without label
            selectedLabel = 0
            centroid = (int(self.output_size[0]/2), int(self.output_size[1]/2), int(self.output_size[2]/2))
        else:
            # randomly select of the label's centroid
            selectedLabel = random.randint(1,segShapeFilter.GetNumberOfLabels())
            centroid = seg.TransformPhysicalPointToIndex(segShapeFilter.GetCentroid(selectedLabel))

        centroid = list(centroid)

        start = [-1,-1,0] #placeholder for start point array
        end = [self.output_size[0]-1, self.output_size[1]-1,self.output_size[2]-1] #placeholder for start point array
        offset = [-1,-1,-1] #placeholder for start point array
        for i in range(2):
            # edge case
            if centroid[i] < (self.output_size[i]/2):
                centroid[i] = int(self.output_size[i]/2)
            elif (image.GetSize()[i]-centroid[i]) < (self.output_size[i]/2):
                centroid[i] = image.GetSize()[i] - int(self.output_size[i]/2) -1

            # get start point
            while ((start[i]<0) or (end[i]>(image.GetSize()[i]-1))):
                offset[i] = self.NormalOffset(self.output_size[i],self.sigma[i])
                start[i] = centroid[i] + offset[i] - int(self.output_size[i]/2)
                end[i] = start[i] + self.output_size[i] - 1

        
        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(self.output_size)
        roiFilter.SetIndex(start)
        croppedImage = roiFilter.Execute(image)
        croppedLabel = []
        for i in range(depth_dim):
                croppedLabel.append(roiFilter.Execute(label[i]))

        return {'image': croppedImage, 'label': croppedLabel}

    def NormalOffset(self,size, sigma):
        s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
        return int(round(random.choice(s)))


