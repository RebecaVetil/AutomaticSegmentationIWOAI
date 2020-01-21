# AutomaticSegmentationIWOAI



## Presentation of the repository
This repository gathers the code of my Master's Thesis in Biomedical Engineering, completed at Imperial College London from May to September 2019.

This code performs automatic segmentation of knee IRM. The data has been provided by the OAI, for the IWOAI 2019 Automatic segmentation Challenge.

## Final Report

This .md version is not completed (yet to do: insert links, insert figures, update equations...) Please see the full and final report [here](https://drive.google.com/file/d/1Hh87_bTS0XDZjWNB4dF3JFDYhXwik4OH/view?usp=sharing).

### Abstract 
Introduced in 2015, the U-Net progressively became the standard starting-point algorithm to perform Medical Images Segmentation. Several publications adapted it to their particular task, tweaking the architecture in very specific ways. This resulted in \textit{hacked} U-Nets which overfit the task or even the dataset. Regarding the Magnetic Resonance Imaging field, this is all the more problematical as MRI protocols generate very different images. Reimplementing these algorithms often turns out to be disappointing and the state-of-the-art of medical imaging segmentation does not improve. Consequently, it has been argued that less attention should be drawn to network design, but rather to the steps surrounding the training procedure. Based on these observations, the present work details our participation in the International Society on Osteoarthritis Imaging Segmentation Challenge which is focused on automated cartilage and meniscal segmentation. This work includes a segmentation pipeline that reimplements original state-of-the-art networks (U-Net, V-Net, Cascaded V-Net) and focuses on their metaparameter and hyperparameter configuration. We undertake interpretable comparisons to evaluate methods that meet challenges inherent to biomedical images, such as extreme class imbalance and huge 3D data volumes. The full implementation (based on Tensorflow) is available in this repository.

Key-words: *U-Net, MRI, biomedical image segmentation, training procedure, class imbalance, deep learning, neural network*.

### I. Introduction
One of the leading cause of disability worldwide, osteoarthritis **(OA)** represents important human and economic burdens across the globe, in particular in high-income countries \cite{bib:Arthritisbynumber}. Since OA affects the entire joint, an accurate and objective assessment of the disease status requires the quantitative imaging of all the tissues involved. Magnetic Resonance Imaging **(MRI)**, thanks to its multiple contrast methods, enables excellent depictions of soft tissues and is therefore a powerful non-invasive tool to increase the understanding of OA. Current studies in OA relie mainly upon whole-knee MRI scans that need to be segmented. Although essential, manual segmentation is time-consuming and is subject to [high inter and intra-specialist variability](https://onlinelibrary.wiley.com/doi/full/10.1002/jmri.22478).

Given both the concern of OA and the technological advances, computed-aided methods for automated segmentation appear as promising solutions to support specialists in detection, diagnostic and analysis. For this reason, the International Society on Osteoarthritis Imaging **(ISOAI)** organises annually a Segmentation Challenge focused on automated cartilage and meniscal segmentation. The participants have to generate fully or semi-automated segmentation routines for 3D knee images, using DESS images from the Osteoarthritis Initiative **(OAI)**. The dataset is made of 84 individuals scanned over two timepoints with manual segmentations available for femoral cartilage, patellar cartilage, tibial cartilage (medial and lateral), and the meniscus (medial and lateral). Some samples are shown in Fig. \ref{fig:Segmentations}.

TODO Insert figure 

For the 2019 edition, the segmentation challenge has been dominated by Deep Learning **(DL)** algorithms that reimplemented state-of-the-art Convolutional Neural Networks **(CNNs)** such as the U-Net, the V-Net, DenseNet or ResNet. Interestingly, the results showed that all the architectures yielded very similar results. This corroborates the authors of the nnU-Net, "not new U-Net" \cite{bib:nnUNet} who believed that too much importance was given to U-Net's architectural aspects instead of focusing on steps that, according to them, really influenced the performances (pre-processing, training, ensembling, post-processing). 

These observations stress the importance of the metaparameters (training procedure) and hyperparameters on automated segmentation. The present work aims to explore the impact of such parameters when training a biomedical image segmentation algorithm.

In this work, we implement, on Tensorflow, a segmentation pipeline which enables to perform automatic segmentation from three state-of-the-art algorithms: the U-Net (2D), the V-Net (2D or 3D) and a Cascaded V-Net (3D). The original architectures of these networks are kept, enabling the user to focus on the configuration of metaparameters and hyperparameters. Working on the IWOAI Challenge, we tackle the two main difficulties of this segmentation task: the big data volume (3D volumes made of 160 images) and the detection of thin, small structures (\textit{e.g.,} the lateral tibial cartilage).

### II. Background
In this section, we recall the essential theoretical concepts and techniques that have been developed so far to perform medical image segmentation using Deep Learning.
 
#### 1. Automatic Segmentation
The biomedical imaging field has widely benefited from automated visual analysis tools that help specialists work faster and more efficiently. Today, it has two main strategies to achieve automated segmentation: the first one, introduced in \cite{bib:UNet} with the U-Net, uses Fully Convolutional Networks **(FCNs)** and achieves, at this point, the best results in terms of accuracy and computational efficiency. The end-to-end approach, the whole-image-at-a-time learning process and the simplicity (no pre/post-processing) of these networks constitute their main assets.  
The second strategy is pixel-based, \textit{i.e.} it generates patches from the images to input them into the Convolutional Neural Networks **(CNNs)**. However, these patches tend to slow down the segmentation as the network has to be run independently for each patch. In addition, they blind the network from global information, hence conveying less semantic context. Consequently, this technique is being progressively abandoned or combined with the first one, as the Cascaded Network presented in Section \ref{subsec:methods_network}. 

#### 2. Convolutional Neural Networks (CNN)
A CNN is a particular type of neural network, \textit{i.e.}, a hierarchical stacking of simple functions (also called layers) that aims to make a more complex function. As they use sequences of convolutional layers followed by non-linear activation functions, CNNs maintain the spatial structure of the input. Thus, each layer:

- Accepts a volume of size $W_1 \times H_1 \times D_1$.
- Has **K** filters, where each filter:
  - Produces an activation map,
  - Operates a convolution on patches of the image that have size $[\textbf{F} \times \textbf{F}]$ and always extend the full depth of the input volume,
  - Slides over the images with an interval of **S** pixel (called the \textit{stride}, it can be seen as the resolution),
  - Can add **P** columns and rows of zeros on the borders of the input \textit{(zero-padding)} to keep the original size of the input,
  - Introduces $F \times F \times D_1$ weights (parameters sharing).
- Produces a volume of size $W_2 \times H_2 \times K$ (where $W_2$ and $H_2$ depend on the spatial extent F and resolution S of the filters)
- Introduces $(F \times F \times D_1) \times K$ weights.

By stacking these layers, the network ends up learning the hierarchy of the filters. The CNNs also contain pooling (downsampling) and fully-connected layers, usually at the very end of the network.

The key ideas behind CNNs come from the characteristics of natural signals: local connections (since filters complete convolution with patches of the image, the spatial structure is maintained), shared weights and pooling, which provides a summary statistic of the nearby outputs. The major advantage of this architecture is that it has large learning capacities, it is easier to train than standard feed-forward neural networks (with same sized-layers), and it does not require implementation of a feature-extraction step as this is automatically done by the network.

#### 3. FCN
As described by \cite{bib:FCN}, FCNs tackle semantic segmentation by considering this task as a trade-off between semantics and location: global information resolves the \textit{what} while local information resolves the \textit{where}. FNCs' architecture follows this idea since they are made of two paths separated by a bottleneck:
- First, a contracting path captures the context of the input image to perform the segmentation: it increases the \textit{what} and reduces the \textit{where}, hence producing coarse ouputs. 
- Second, an expansive path localizes precisely the contextual information obtained from the first path. 
While the contracting path has the typical architecture of a CNN, the expansive path is made of upsampling layers (also called \textit{deconvolution} layers) that enable recovering the spatial resolution. However, as concatenating these two paths is not enough to produce fine-tuned outputs, skip connections (also denoted as \textit{long} skip connections in \cite{bib:SkipConnections}) are added to the network. These connections combine higher layers (high-level semantic information, low spatial resolution) with lower layers (low-level semantic information, high spatial resolution), hence producing a high resolution segmentation map.

### Methods
The segmentation routine presented in this work makes use of different methods to address some challenges inherent to biomedical images. In particular, it implements several state-of-the-art network architectures, several loss functions as well as some data augmentation techniques. These implementations are described hereinbelow.

*In this section, a convolution is referred to as $C_{U}^{F \times F}(c_{in}, c_{out})$ where $F$ is the filter size, $U$ means unpadded convolution, $c_{in}$ (respectively $c_{out}$) is the number of channels of the input (respectively the ouput), and $n_{conv}^{i}$ is the number of convolutional layers of the level $i$ . $C^T$ refers to a deconvolution layer and $B,P,D$ stands for the stacking of Batch Normalization, Parametric Leaky Relu and Dropout Layers}.*

#### 1. Addressing the big volume of data with different network architectures}
Big data volumes are common in biomedical imaging since most of the scans are 3D. For instance, each MR scan of the IWOAI challenge is composed of 160 slices of size $384 \times 384$. Consequently, processing and learning from the whole volume appears to be very computationally expensive and can impede some learning processes. To cope with big volumes, three different network architectures have been implemented in our data pipeline: a U-Net, a V-Net, and a cascaded V-Net.
These algorithms can be run from the terminal, using the following command: 
`python train.py --data_dir path/to/data`


**U-Net**
A first simple but widely used method consists in unstacking the 3D data to analyse it in 2D. Therefore, we implement a U-Net, which has been introduced by \citeauthor{bib:UNet} and still stands as the starting-point FCN to perform medical images segmentation.

The architecture of our U-Net is very similar to the original one and consists of:
    \item A contracting path made of 4 levels $L_i^{c}$, each level doing:
    \begin{enumerate}
        \item Double the number of feature channels: [$C_{U}^{3 \times 3}(c_{in}, 2 \times c_{in}) + B,P,D$ ]
        \item Compression block: [$C_{U}^{3 \times 3}(c_{in}, c_{in}) + B,P,D$] $\times n_{conv}^{i}$
        \item Feature storing: store the tensor as fine-grained features for the expansive path
        \item Downsampling: [${maxpool}^{2 \times 2}$ with strides $2 \times 2 + B,P,D$]
    \end{enumerate}
    \item A bottleneck:
    \begin{enumerate}
        \item Convolve: [$C_{U}^{3 \times 3}(c_{in}, c_{in}) + B,P,D$] $\times 2$
    \end{enumerate}
    \item An expansive path made of 4 levels $L_i^{e}$, each level doing:
    \begin{enumerate}
        \item Upsampling: double the spatial dimensions, halve the channels [$C^{T, 2 \times 2} (c_{in}, c_{in}/2) + B,P,D$]
        \item Concatenate the fine-grained features from $L_i^{c}$: [concatenation $+ B,P,D$] 
        \item Decompression block: [$C_{U}^{3 \times 3}(c_{in}, c_{in}) + B,P,D$] $ \times n_{conv}^{i}$
    \end{enumerate}
    \item Output:
    \begin{enumerate}
        \item Mapping each feature vector to the desired number of classes: [$C_{U}^{1 \times 1}(c_{in}, n_{classes}) + B,P$ ]
    \end{enumerate}
\end{itemize}

Total number of convolutional layers: 23

Total number of parameters: 2396544

This network can be run using the following parse arguments: 
 `--seg_type 'VNET'`

**V-Net**
We also implement the V-Net, an adaptation of the U-Net for 3D volumes which has been first introduced in \citeauthor{bib:VNET}. If the general principle remains similar, the main difference lies in the amount of memory required. Again, the architecture of our V-Net is very similar to the U-Net one, apart from the following elements: 
- Each operation is adapted to process 3D input data
- Each convolution filter has size $5 \times 5 \times 5$
- The number of output channels in the first level is set to 16 (\textit{vs} 64 for the U-Net)
- The network uses residual learning to speed up the convergence: at the end of the last compression (or decompression) block of each level, the input of this very level goes through non-linearities and is then summed to the output of the last convolutional layer.
- The maxpooling operations are replaced by $2 \times 2 \times2$ convolutional layers that halve the dimensions of the feature map.
To be able to assess the impact of these architectural changes compared to the 2D U-Net, this V-Net can be run on 2D images using the following parse arguments: 
`
    --seg_type 'VNET' --data_dim '2D' `
 (\verb!data_dim! should be set to \verb!‘3D’! for a 3D V-Net task). If the V-Net makes up the best method in an ideal world, it is limited by the GPU memory and can not be used on big volumes of images.

**Cascaded V-Net**
\textit{Here, we refer to the batch size as $N$, the original (respectively modified) height/weight/depth of the MRI scans as $H/W/D$ (respectively as $h/w/d$), the number of channels as C, the number of masks as $M$, the MRI inputs as $I$, the true segmentations as $S^*$ and the predicted segmentations as $S$.}

To address the downsides of the V-Net but still making use of its powerful semantic understanding, Cascaded Networks have been proposed in several works \cite{bib:nnUNet}. Such networks work by stages at different resolutions, and we build ours in the following way:
\begin{enumerate}
    \item \textit{Low resolution stage}: we run a 3D V-Net on low resolution images (cf Fig. \ref{fig:Downsampling})
    \begin{itemize}
        \item First, the MRI scans and the corresponding segmentations get their spatial dimensions divided by $2^{D}$, where \verb!D!, the down factor, is set to $2$ for our experiments. 
        \item Then, these low resolution inputs $I_{LOW}(N,h,w,D,c)$ and $S_{LOW}^*(N,h,w,D,m)$ are fed to the 3D V-Net.
        \item The outputs of the network are low resolution segmentations $S_{LOW}(N,h,w,D,m)$.
        \item Finally, the network is trained to match $S_{LOW}^*$ on x epochs.
    \end{itemize}
    \item \textit{Patch stage}: we run a 3D V-Net on patches. (cf Fig. \ref{fig:Patching})
    \begin{itemize}
        \item First, the full resolution MRI scans $I_{FULL}(N,H,W,D,c)$ are passed through the low resolution network.
        \item The resulting low resolution segmentations are then upsampled to the original resolution  $S_{LOW}^{UP}(N,H,W,D,m)$, and added as additional channels to $I_{FULL}$, to give $I_{ADD}$.
        \item Then, $I_{ADD}(N,H,W,D,c+m)$ and $S_{FULL}^*(N,H,W,D,m)$ are broken down into $n$ patches of size $[patch\_size \times patch\_size]$. These new inputs, $I_{PATCH}(n,h^-,w^-,D,c+m)$ are fed to a new 3D V-Net. 
        \item The outputs of the network are patched segmentations $S_{PATCH}(n,h^-,w^-,D,m)$. The full resolution segmentation is then reconstructed to obtain $S_{FULL}(N,H,W,D,m)$.
        \item Finally, the network is trained to match $S_{FULL}^*$ on x epochs.
    \end{itemize}
\end{enumerate}
In brief, the first V-Net learns to segment roughly from the whole image. Then, the outputs of this wide-angle network are transmitted and used by the second V-Net which learns to segment more finely but from patches only. This can be run using the following parse arguments: 
\begin{verbatim}
    --seg_type ‘cascade’--data_dim '3D'    
\end{verbatim}

TODO Insert figure 

TODO Insert figure 

#### 2. Addressing the class imbalance with different loss functions}
Some biomedical segmentation tasks present high class imbalance, specially when it comes to the segmentation of very precise and small structures. Consequently, some classes can be highly under-represented, making their recognition difficult. This is all the more challenging when it comes to 3D segmentation since some slices (often, the first and the last) may not show any label at all, \textit{i.e.} are composed entirely of background. Regarding this matter, the IWOAI dataset is a great example of class imbalance: first, the femoral cartilage, the patellar cartilage, the tibial cartilage and the meniscus are thin tissues that make up very small proportions of the whole image. As shown in Fig. \ref{fig:Hist}, most of the scans for this challenge are made of background (\textit{i.e.} of tissues that are not being segmented): the tissues that have to be segmented stand for as little as \textbf{0.588\%} (\tikz\draw[magenta,fill=magenta] (0,0) circle (.5ex); Femoral Cartilage), \textbf{0.105\% }(\tikz\draw[red,fill=red] (0,0) circle (.5ex); Medial Tibial Cartilage), \textbf{0.094\%} (\tikz\draw[green,fill=green] (0,0) circle (.5ex); Lateral Tibial Cartilage), \textbf{0.118\%} (\tikz\draw[blue,fill=blue] (0,0) circle (.5ex); Patellar Cartilage), \textbf{0.111\% }(\tikz\draw[cyan,fill=cyan] (0,0) circle (.5ex); Lateral Meniscus) and \textbf{0.107\%} (\tikz\draw[yellow,fill=yellow] (0,0) circle (.5ex); Medial Meniscus), when the background makes up \textbf{98.876\%} of the whole training dataset.

TODO Insert figure 

Thus, the frequency of each label makes more sense when visualised per slice, as in Fig. \ref{fig:HistPerSlice}. Interestingly, we can see that the frequency curves present a symmetry around the middle of the depth-dimension (slice 80). This is because the scans were acquired in different anatomical directions (posterior to anterior or \textit{vice versa}) and this characteristic is actually a great asset of the OA dataset since it prevents the networks from learning a bias due to the scanning direction. 


TODO INSERT FIGURE 

However, even the analysis by slice (Fig. \ref{fig:HistPerSlice}) indicates that the dataset is highly unbalanced and this may lead the classifier to only predict background. Consequently, in order to ensure a proper and efficient learning, the loss function has to be chosen carefully and our training pipeline offers different options for this.

*We use the following notations:*
    \item $n_{ij}$: number of pixels of class $i$, predicted to belong to class $j$
    \item $p_{pix,i}$ ($p_{pix,i}^*$): predicted (true) probability for pixel \textit{pix} to belong to class $i$
    \item $n_{CL}$: number of different classes
    \item $t_i = \sum_j n_{ij}$: number of pixels of class $i$
    \item $n_{tot}$: total number of pixels

- **Cross Entropy Loss**
For each individual pixel, this loss compares the true and the predicted class and the result is then averaged over all the pixels: 
\begin{equation} \label{eq:xent}
 CE = - \frac{ \sum\limits_{pix=1}^{n_{tot}} \left(  \sum\limits_{i=1}^{n_{CL}} p_{pix,i}^* \times log(p_{pix,i}) \right) }{n_{tot}}
\end{equation}
Even if this loss stands among the most commonly used, its power is highly limited for unbalanced problems since the prevalent class will totally dominate the loss value.

- **Weighted Cross Entropy**
This loss aims to address the unbalanced representation of some classes and was first discussed in \cite{bib:FCN}. 
\begin{equation} \label{eq:wxent}
 wCE = - \frac{ \sum\limits_{pix=1}^{n_{tot}} \left(  \sum\limits_{i=1}^{n_{CL}}\omega_i \times  p_{pix,i}^* \times log(p_{pix,i}) \right) }{n_{tot}}
\end{equation}
For our specific problem, we compute it by setting manually the weight of the background class $\omega_{background}$ to \verb!0.5!, and to \verb!1.0! for all the other classes. Then, the Cross Entropy Loss value of each pixel is multiplied by these weights, based on their ground-truth class: as a consequence, the background pixels contribute much less to the averaged loss value.
The background weight has been defined as a hyperparameter that can be easily modified using the \verb!background_weight! parse argument. According to \cite{bib:FocalLoss}, this value should be defined using cross validation.

- **Jaccard index**
The Jaccard index (also known as Intersection-Over-Union) ranges from 0\% (worst case) to 100\% (perfect case) and is computed as the ratio between the intersection (overlap) and the union:
\begin{equation} \label{eq:jaccard}
IoU = \left\langle \frac{target \cap pred}{target \cup pred} \right\rangle_{CL}= \frac{1}{n_{CL}} \times  \sum\limits_{i=1}^{n_{CL}} {\frac{n_{ii}}{t_i + \sum\limits_{j=1}^{n_{CL}} n_{ji} - n_{ii}}}
\end{equation}
This index enables working with classes that are not very represented within the image since common activations are normalized by the number of activations in each image individually. 
In order to get a loss function that can be minimised, we use $1 - IoU$.

- **Sorensen-Dice coefficient**
The Sorensen index (or Dice Coefficient, or F1 score) is an overlap measure which is very similar to the Jaccard Index:
\begin{equation} \label{eq:sorensen}
F1 = \left\langle \frac{2 \times target \cap pred}{target + pred} \right\rangle_{CL} = \frac{2}{n_{CL}} \times  \sum\limits_{i=1}^{n_{CL}} {\frac{n_{ii}}{t_i + \sum\limits_{j=1}^{n_{CL}} n_{ji}}}
\end{equation}
Again, we use $1 - F1$.

- **Focal**
The  Focal Loss has been introduced in \cite{bib:FocalLoss} to address extreme class imbalance. Inspired from the Cross Entropy, it down-weights the loss values obtained by correct classifications and, consequently, the easy predictions (\textit{e.g.}, background predictions) do not overwhelm the training. Where the weighted Cross Entropy discriminates background \textit{vs} not background predictions, the  Focal Loss rather focuses on easy \textit{vs} hard predictions. This is done thanks to a focusing parameter $\gamma$:
\begin{equation} \label{eq:focal}
 FL = - \frac{ \sum\limits_{pix=1}^{n_{tot}} \left(  \sum\limits_{i=1}^{n_{CL}} (1-p_{pix,i})^{\gamma} \times log(p_{pix,i}) \right) }{n_{tot}}
\end{equation}
As we can see, the higher the $\gamma$, the more the loss is unaffected by an easy classification (with high $p_{pix,i}$).

- **Combined**
This loss is directly inspired from the nnU-Net \cite{bib:nnUNet} and is simply the sum of the Sorensen and Cross Entropy losses. 
\begin{equation} \label{eq:combined}
L_{combined} = F1 + CE
\end{equation}

- **Weighted combined**
This loss is a variant of the Combined Loss as it is the sum of the Sorensen and weighted Cross Entropy.
\begin{equation} \label{eq:wcombined}
L_{wcombined} = F1 + wCE
\end{equation}
Any of these losses can be chosen using the following parse argument:
\begin{verbatim}
    --loss_function 'name_of_the_loss'
\end{verbatim}

#### 3. Pre-processing and data augmentation
Finally, the last noteworthy method used to tackle the challenge is data augmentation. Thus, the segmentation pipeline offers various transformations that can be applied to the images, such as normalisation (automatic, statistical, or manual normalisation), intensity inversion, random noise or crop (random or confidence). Fig. \ref{fig:DataAugmentation} shows the effects of random crop (with a final size of $[250 \times 250]$) and manual normalisation (setting the intensities values of the output image between \verb!0! and \verb!0.005!).

### IV. Results and experiments
#### 1. Material
Two datasets are provided: training (60 patients) and validation (14 patients). Each patient, for any of these sets, has two 3D scans corresponding to two timepoints (V00 and V01) (Fig. \ref{fig:Segmentations}). For each version, we have the grayscale image scan, of shape $[384, 384, 160]$ and type float32, and the manually segmented scan, of shape $[384, 384, 160,6]$ and type uint8. The 4-th dimension of the segmentation scans corresponds to the six masks we have to segment:
- \tikz\draw[magenta,fill=magenta] (0,0) circle (.5ex); Femoral Cartilage, 
- \tikz\draw[red,fill=red] (0,0) circle (.5ex); Medial Tibial Cartilage,
- \tikz\draw[green,fill=green] (0,0) circle (.5ex); Lateral Tibial Cartilage,
- \tikz\draw[blue,fill=blue] (0,0) circle (.5ex); Patellar Cartilage,
- \tikz\draw[cyan,fill=cyan] (0,0) circle (.5ex); Lateral Meniscus and 
- \tikz\draw[yellow,fill=yellow] (0,0) circle (.5ex); Medial Meniscus. 

The data is stored with the h5 format, but, for more convenience, we convert it into Numpy arrays and add the background as an additional mask. For visualisation matters, all the masks are displayed on a same image (Fig. \ref{fig:Segmentations}b) and with a color code (Fig. \ref{fig:Segmentations}c).

4 RTX6000 GPUs were used for the training.

#### 2. Description and results of the different experiments}
To avoid any confusion, different formats are used for \textit{loss functions} and \verb!metrics!.

In order to assess the effects of the different architectures and losses, we train several networks with different loss functions. After each epoch, we test each network on the validation set by computing the following metrics: \verb!Accuracy!, \verb!Focal Loss!, \verb!Sorensen Loss!, \verb!Jaccard Loss! (defined in section \ref{subsec:methods_losses}). These metrics are averaged over the validation set for each of the 12 experiments. These experiments fall into the following comparisons:

**A. 2D Network comparisons**

Using a Cross Entropy Loss function and data augmentation, we train the following networks:
- U-Net 
- 2D V-Net 
A *weighted Combined Loss* function and data augmentation were used for these experiments. Fig.\ref{fig:NetworkComparison}, in the Appendix, shows that, regardless of the metric, the U-Net performs slightly better than the 2D V-Net and the extreme values (mininimum and maximum) remain better for the U-Net during all the training procedure. However, the learning curves are very similar and both networks reach, after 10 epochs, the same average performances. Moreover, the V-Net appears to be more computationally efficient than the U-Net as the V-Net completes the 10 epochs in $20.10^3$ seconds, when the U-Net is 1.4 times slower.


**B. 3D Network comparisons**
- 3D V-Net 
- Cascaded V-Net
This comparison was impeded by the amount of memory of our hardware configuration (made of 4 GPUs, with 16 CPUs of 96 GB each).

**C. Loss comparisons**
We train 2D U-Nets with data augmentation, using the following loss functions:
\begin{enumerate}
    \item \textit{Combined Loss} 
    \item \textit{Focal Loss}, \verb!g! $= \gamma = 1$ (focusing parameter) 
    \item \textit{Focal Loss}, \verb!g! $= \gamma = 2$ 
    \item \textit{Jaccard Loss}
    \item \textit{Sorensen Loss}
    %\item \textit{Weighted Combined Loss} %UNET_gpu_transf.pbs, waiting
    \item \textit{Weighted Cross Entropy Loss}, \verb!w = 0.5! (weight of the background)
    \item \textit{Weighted Cross Entropy Loss}, \verb!w = 0.2! 
    \item \textit{Cross Entropy Loss} 
\end{enumerate}


Beginning the analysis with Fig. \ref{fig:LastEpochComparison}, one can observe that the performances of the network trained with the \textit{Cross Entropy Loss} and the \textit{Combined Loss} are almost identical. This suggests that, for the \textit{Combined Loss}, the training is largely driven by the cross entropy term. These networks stand among the best for all the \verb!metrics!, except regarding the \verb!focal loss metric! for which the \textit{ Focal Loss functions} definitively outshine. Regarding the focusing parameter, \verb!g! $= \gamma = 1$ seems to outperform \verb!g! $= \gamma = 2$ for all the metrics except the \verb!Focal Loss! one.

Unsurprisingly, Fig. \ref{fig:LossComparison} shows that the \verb!Sorensen! and \verb!Jaccard! metrics are positively correlated for all the networks. In particular, the \textit{Focal Loss functions} achieve the worst results and show room for improvement for these two metrics. Similarly, the networks trained with the \textit{Sorensen} and \textit{Jaccard Loss functions} perform poorly regarding the \verb!Focal Loss metric!. More generally, the \textit{Jaccard}, and especially the \textit{Sorensen network}, show low accuracy performances on this challenge.
Last, the two networks trained with \textit{weighted Cross Entropy Loss functions} yield results similar to the \textit{normal Cross Entropy Loss network}, making the relevance of the weighting difficult to assess when considering only the \verb!metrics!. Here again, the value of the background weight does not impact significantly the results, making the hyperparameter search tough.

Following with time efficiency analysis, Fig. \ref{fig:TrainingTime} shows the training time for each network.  This time, the two \textit{focal networks} show different characteristics as the \verb!g! $= \gamma = 1$ was nearly 1.1 times slower. Also, it appears that the \textit{Combined Loss network} completes the 10 epochs with the best training time, outperforming networks whose loss functions involve \textit{cross entropy}.

The analysis can be completed with a more empirical, but not less essential, assessment based on the visualisation of the predicted masks. Segmentation results for random slices from the validation set are displayed on Fig. \ref{fig:Pred2} (and Fig. \ref{fig:Pred0} and \ref{fig:Pred1} in the Appendix). First, we can see that, contrary to what was suggested by the metric analysis, the "normal" cross entropy is visually outperformed by the combined or weighted cross entropy losses. Very interestingly, the Combined Loss shows outstanding results, while Fig. \ref{fig:LastEpochComparison} implies that it does not beat the normal Cross Entropy Loss. Last, if the visualisation does not allow to decide which focusing parameter is the best between 1 and 2 for the Focal Loss, it suggests that the weighted cross entropy is more accurate for \verb!w = 0.5!.

\textit{Note: networks for experiments A and C are trained on the 160 slices of the two timepoints scans for the 60 individuals, \textit{i.e.} on 19 200 slices. The training is performed using the Gradient Descent Optimiser, with a batch size of 1 and a learning rate set to} \verb!1e-2! \textit{Data Augmentation transformations refer to (Manual Normalisation (clipping values between 0 and 0.005) and random crop to $[300 \times 300]$)}.

### V. Discussion

First and foremost, the segmentation task has been successfully completed since the 2D networks, whether they are V-Nets or U-Nets, achieve very respectable results and this despite of the 3D nature of the problem. 

The pipeline, with its easy parse argument configuration, enables to cleverly tweak the training procedure in order to easily achieve hyperparameter and metaparameter search. In addition to enable the exploration of new loss functions, the pipeline also permits to validate results from previous works. For instance, experience A (cf \ref{subsec:methods_network}) confirms \citeauthor{bib:SkipConnections} regarding the effects of skip connections on the training time, while experience C confirms the relevance of the \textit{Combined Cross Entropy Loss function,} introduced by \citeauthor{bib:nnUNet}.
However, and even if the results are satisfying, further research could be carried out in order to improve the segmentation results.
First, one could think of a different training strategy in order to make the most of the big number of training sets available. Indeed, as suggested by \citeauthor{bib:UNet} who reached outstanding segmentation results with only 30 images, 120 training volumes represent a huge number of data. Training a network on the whole dataset for \textit{x} epochs, without any change in the parameter configuration, can lead to a plateau in the learning curve, as observed in Fig. \ref{fig:LossComparison}. To fully harness the semantic information and avoid overfitting, hyperparameters could have been changed throughout the learning process (following the idea of a decade learning rate). This hyperparameters evolution could be done following an epoch sequence (\textit{i.e.}, modifying parameter $p_a$ after \textit{n} epochs), and/or following a version sequence (\textit{i.e.}, train V00 data and V01 data separately, with different parameters). Indeed, since there is an obvious link between each V00 and V01 sets, we believe a clever use of these two timepoints may lead to some improvements.

Second, some changes could be brought to our Cascaded Network in order to make it more feasible from a computational point of view. For instance, the first wide-angle V-Net could be simplified as the rough segmentation may need less levels and less convolutional layers. This first classifier could also be binary and only detect the background. By doing so, the last dimension of the input would be of 3 instead of 8, hence reducing the amount of memory required. Another strategy could consist in randomly downsampling the 3D volume in the depth dimension, \textit{i.e.} only working on a selection of slices. All these memory savings could be exploited to increase the size of the patches, which should help the segmentation.

Third, more tests could be done regarding data augmentation and pre-processing. As suggested by the literature review, these steps can drastically improve the segmentation results and should be extensively used. Even if several transformations were implemented in our segmentation routine, the experiments only used random crop and manual normalisation. Thus, more transformations could be implemented and tested, notably random elastic deformations which are commonly used in medical imaging.

Finally, the validation step of this work could be improved as the comparison of the different losses turned out to be very tough and uncertain. Metrics like the \verb!Accuracy!, the \verb!Sorensen Loss! or the \verb!Jaccard Loss! appeared to be insufficient for several reasons. Indeed, different training procedures achieved very similar scores despite of significant visualisation differences. More specifically, the analysis shows that good scores do not necessarily go along with good visualisation results, and \textit{vice versa}. Thus, even if these metrics give some useful insights about the general performances of a network, they do not enable to establish a straight-forward, unequivocal ranking between the different experiments. Finally, the \verb!Focal Loss metric!, thanks to its focusing parameter $\gamma$, is the one that favours more complex predictions, giving more importance to the learning of femoral or meniscus classes. These results are consistent with the visualisation results. Therefore, among the different loss functions studied, both \textit{Focal Losses functions} have the best segmentation performances. Yet, this conclusion should be handled carefully as a more comprehensive evaluation, with metrics computed by class and error histograms, should be carried out.

### V. Conclusion
In this work, we have built a segmentation pipeline that makes use of different strategies in order to perform efficient and accurat segmentations of biomedical images. In addition to achieve honorable segmentation results for the IWOAI Segmentation Challenge, this paper brings to light general takeaways that should be kept in mind when performing biomedical image segmentation:

First, this work verifies that 3D segmentation problems are quickly limited by the amount of hardware memory required. By using patches combined with low resolution information, Cascaded Networks can be a solution. Nevertheless, they are still limited by huge data volumes such as the ones in the IWOAI challenge. Therefore, transforming the problem into 2 dimensions often makes up the best solution.

Second, this work emphasizes the necessity of a well-thought parameter search during the training of a deep learning algorithm. As recommended in this paper, sticking to the original state-of-the-art U-Net and focusing, first on the metaparameters and second, on the hyperparameters, appears to be a good strategy. 

Third, this work underlines the extreme importance of an accurate and robust validation metric to avoid misleading metrics. This is even more crucial when dealing with highly unbalanced problems, specially when the labels of interest are in minority. Indeed, computing an interpretable metric that cleverly reflects the expected response appears to be essential to value a network. As discussed in section \ref{sec:discussion}, the Focal Loss metric manages to capture the performance regarding the valuable information.

Last but not the least, this paper shows that highly unbalanced datasets can be addressed with a careful choice of the loss function. By compelling it to look more specifically to the very under represented classes, \textit{i.e.} cartilage and meniscus, we impede the network to systematically predict the easy class, \textit{i.e.} the background. This can be done with weighting or focusing parameters as for Weighted Cross Entropy or Focal Losses. However, results showed that honourable performances were also achieved by the combination of two non-parametric losses, the Cross Entropy and the Sorensen ones. Future work could therefore explore new ways of combining functions to leverage the assets of different losses.

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
