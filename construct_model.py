# %%
import tensorflow as tf
from yolo_layer_utils import Predict
import numpy as np
import os
import xml.etree.ElementTree
from rich.console import Console
from typing import List
from tqdm import tqdm

#%%

# ==========================================================================
#                     🔵 Mixin for Anchor Generation
#       It can be used to inject Anchor boxes generation
#                   functionality into any child class
# ==========================================================================



# %%

# ==========================================================================
#                  common interface for all YOLO models
#  it will be holding the core functions for building all the model
# ==========================================================================
class IYoloFamily:
    initializer = tf.random_normal_initializer(stddev=0.01)
    l2 = tf.keras.regularizers.l2(4e-5)

    def __init__(self) -> None:

        # following will be set later
        self.class_dict = None
        version = None
        # anchor boxes computed over the dataset , these will be computed at runtime
        # self.anchors = np.array([[17.0, 21.0], [24.0, 51.0], [41.0, 100.0],
        #                          [45.0, 31.0], [75.0, 61.0], [94.0, 129.0],
        #                          [143.0, 245.0], [232.0, 138.0], [342.0, 299.0]], np.float32)
        self.max_boxes = 150
        self.versions = ['s', 'm', 'l', 'x']
        self.width = [0.50, 0.75, 1.0, 1.25]
        self.depth = [0.33, 0.67, 1.0, 1.33]
        self.image_size = 640
        # following parameters aare used for building the different models for YOLO
        # depth = self.depth[self.versions.index(version)]
        # width = self.width[self.versions.index(version)]

    def conv(self, x, filters, k=1, s=1):
        '''
        comvolutional block needed for building the model

        Parameters
        ----------
        x : int
            input tensor
        filters : int
            no. of filters to be applied
        k : int, optional
            kernel size for convolution, by default 1
        s : int, optional
            stride for convolution, by default 1

        Returns
        -------
        tf.Tensor
            output tensor after application of conv. block
        '''
        if s == 2:
            x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            padding = 'valid'
        else:
            padding = 'same'
        x = tf.keras.layers.Conv2D(filters, k, s, padding, use_bias=False,
                                   kernel_initializer=IYoloFamily.initializer, kernel_regularizer=IYoloFamily.l2)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.03)(x)
        x = tf.keras.layers.Activation(tf.nn.swish)(x)
        return x

    def residual(self, x, filters, add=True):
        inputs = x
        if add:
            x = self.conv(x, filters, 1)
            x = self.conv(x, filters, 3)
            x = inputs + x
        else:
            x = self.conv(x, filters, 1)
            x = self.conv(x, filters, 3)
        return x

    def csp(self, x, filters, n, add=True):
        y = self.conv(x, filters // 2)
        for _ in range(n):
            y = self.residual(y, filters // 2, add)

        x = self.conv(x, filters // 2)
        x = tf.keras.layers.concatenate([x, y])

        x = self.conv(x, filters)
        return x

    def build_model(self):
        raise NotImplementedError


# Concrete model maker class
class Yolo(IYoloFamily):

    # override the method and change the arguments
    def build_model(self, 
                    version: str, 
                    n_classes: int, 
                    anchors : np.ndarray,
                    training: bool = True):
        '''
        overridden method for building the model, model building generally is done after dataloder is created because
        no. of classes is needed for building the model. So it assumes that a dictionary mapping for class names to integer 
        labels is already created and n_classes is known.
        
        Such mapping should come from the dataloader. This method will only build different variants of the yolo family model
        
        ⚠️ Here is the order of execution:
        1) Anchor generation (will output anchor boxes)
        2) Dataloader creation (will output class labels to integer mapping and will also need anchors generation in above step)
        3) Model building (will need class labels to integer mapping and anchors generation in above steps)
        
        
        Note
        ------
        ☠️ class labels to integer mapping should be consistent throughout training process. If such mapping is calculated inside model building
        method then it will be different everytime model is built and it will lead to inconsistency in training and testing process.


        Parameters
        ----------
        version : str
            can be 's', 'm', 'l', 'x'
        n_classes : int
            total no. of classes in the dataset
        anchors : np.ndarray, optional
            anchor boxes for the dataset, (default computed on VOC) 
        training : bool
            can be True or False, if True then it will predict feature maps at 3 scales, otherwise it will attach post-processing (NMS) layer at the end

        Returns
        -------
        IYoloFamily
            
        '''        
        self.n_classes = n_classes
        self.version_selected = version
        # following is needed for post-processing in case of evaluation mode
        self.anchors = anchors
        # select depth and with on the basis of the model selected

        
        depth = self.depth[self.versions.index(version)]
        width = self.width[self.versions.index(version)]

        # ==========================================================================
        #                             start building the model
        # ==========================================================================

        inputs = tf.keras.layers.Input([self.image_size, self.image_size, 3])
        #
        x = tf.nn.space_to_depth(inputs, 2)
        x = self.conv(x, int(round(width * 64)), 3)
        x = self.conv(x, int(round(width * 128)), 3, 2)
        x = self.csp(x, int(round(width * 128)), int(round(depth * 3)))

        x = self.conv(x, int(round(width * 256)), 3, 2)
        x = self.csp(x, int(round(width * 256)), int(round(depth * 9)))
        x1 = x

        x = self.conv(x, int(round(width * 512)), 3, 2)
        x = self.csp(x, int(round(width * 512)), int(round(depth * 9)))
        x2 = x

        x = self.conv(x, int(round(width * 1024)), 3, 2)
        x = self.conv(x, int(round(width * 512)), 1, 1)
        x = tf.keras.layers.concatenate([x,
                                         tf.nn.max_pool(x, 5,  1, 'SAME'),
                                         tf.nn.max_pool(x, 9,  1, 'SAME'),
                                         tf.nn.max_pool(x, 13, 1, 'SAME')])
        x = self.conv(x, int(round(width * 1024)), 1, 1)
        x = self.csp(x, int(round(width * 1024)), int(round(depth * 3)), False)

        x = self.conv(x, int(round(width * 512)), 1)
        x3 = x
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.concatenate([x, x2])
        x = self.csp(x, int(round(width * 512)), int(round(depth * 3)), False)

        x = self.conv(x, int(round(width * 256)), 1)
        x4 = x
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.concatenate([x, x1])
        x = self.csp(x, int(round(width * 256)), int(round(depth * 3)), False)
        # ⚠️ P3 --> image_size // 8
        p3 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1,
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2, name='P3_80x80')(x)
        # p3 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1, name=f'p3_{self.image_size//8}x{self.image_size//8}x3x{self.n_classes+5}',
        #                             kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        x = self.conv(x, int(round(width * 256)), 3, 2)
        x = tf.keras.layers.concatenate([x, x4])
        x = self.csp(x, int(round(width * 512)), int(round(depth * 3)), False)
        # ⚠️ P4 --> image_size // 16
        p4 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1,
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2, name='P4_40x40')(x)
        # p4 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1, name=f'p4_{self.image_size//16}x{self.image_size//16}x3x{self.n_classes+5}',
        #                             kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        x = self.conv(x, int(round(width * 512)), 3, 2)
        x = tf.keras.layers.concatenate([x, x3])
        x = self.csp(x, int(round(width * 1024)), int(round(depth * 3)), False)
        # ⚠️ P5 --> self.image_size // 32
        p5 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1,
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2, name='P5_20x20')(x)
        # p5 = tf.keras.layers.Conv2D(3 * (self.n_classes + 5), 1, name=f'p5_{self.image_size//32}x{self.image_size//32}x3x{self.n_classes+5}',
        #                             kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        # ==========================================================================
        #                           ☠️  Model Output layer Surgery
        # ==========================================================================
        if training:
            # ⚡ in training model the prediction will be performed over 3 feature maps
            return tf.keras.Model(inputs, [p5, p4, p3])
        else:
            # ⚡ in inference model the NMS layer will be made a part of the model, the last prediction layer will hold no learnable parameters
            return tf.keras.Model(inputs, Predict(anchors=self.anchors, image_size=self.image_size, n_classes=self.n_classes, max_boxes=self.max_boxes)([p5, p4, p3]))


# common function to construct the model

def construct_model(cls, 
                    version: str, 
                    class_dict : dict, 
                    anchors : np.ndarray = np.array([[19.0, 25.0], [29.0, 61.0], [51.0, 104.0], [58.0, 40.0], [97.0, 179.0], [110.0, 89.0], [179.0, 268.0], [240.0, 143.0], [372.0, 298.0]]), 
                    is_training: bool = True) -> IYoloFamily:
    '''
    this function will be used to build all variants of the yolo family of models

    Parameters
    ----------
    version : str
        can be one of the following: 's', 'm', 'l', 'x'
    n_classes : int
        number of classes in the dataset
    anchors : np.ndarray
        [9, 2] should be the shape of the array, if not provided default anchors calculated on VOC will be used.
    is_training : bool
        if training mode, set it to True otherwise False

    Returns
    -------
    IYoloFamily
        yolo family of models
    '''
    # get instance of the concrete class Yolo
    model_class = cls()
    # If anchors need to be generated
    # model_class._get_info_for_generating_anchors(data_dir="../dataset_validation/",
    #                                             #  class_dict willcome from the dataloader since generally model building needs no. of classes
    #                                             class_dict=class_dict
                                        #  ).generate_anchor()
    model = model_class.build_model(
                                    version=version, 
                                    n_classes=len(class_dict), 
                                    anchors=anchors, 
                                    training=is_training
                                    )

    return model


def main():

    yolo = construct_model(Yolo, 's', {                                                                                                                        
                                        'aeroplane': 0,                                                                                                      
                                        'bicycle': 1,                                                                                                        
                                        'bird': 2,                                                                                                           
                                        'boat': 3,                                                                                                           
                                        'bottle': 4,                                                                                                         
                                        'bus': 5,                                                                                                            
                                        'car': 6,                                                                                                            
                                        'cat': 7,                                                                                                            
                                        'chair': 8,                                                                                                          
                                        'cow': 9,                                                                                                            
                                        'diningtable': 10,                                                                                                   
                                        'dog': 11,                                                                                                           
                                        'horse': 12,                                                                                                         
                                        'motorbike': 13,                                                                                                     
                                        'person': 14,                                                                                                        
                                        'pottedplant': 15,                                                                                                   
                                        'sheep': 16,                                                                                                         
                                        'sofa': 17,                                                                                                          
                                        'train': 18,                                                                                                         
                                        'tvmonitor': 19                                                                                                      
                                    }, 
                                    np.array([[19.0, 25.0], [29.0, 61.0], [51.0, 104.0], [58.0, 40.0], [97.0, 179.0], [110.0, 89.0], [179.0, 268.0], [240.0, 143.0], [372.0, 298.0]]),
                                    True)

    print(yolo.summary())
    random_image = tf.random.normal((8, 640, 640, 3)) # 8 is the batch size
    p5_pred, p4_pred, p3_pred = yolo.predict(random_image)
    breakpoint()
    
    tf.keras.utils.plot_model(
                                yolo,
                                to_file='model_s.png',
                                show_shapes=True,
                                show_dtype=False,
                                show_layer_names=True,
                                rankdir='TB',
                                expand_nested=False,
                                dpi=300,
                                layer_range=None,
                                show_layer_activations=False
                            )

# %%


if __name__ == '__main__':
    main()



#%% 

# import matplotlib.pyplot as plt
# import pandas as pd
# f, ax = plt.subplots(1, 1, figsize=(10, 10))
# pd.read_csv('training_logs.csv', skipinitialspace=True).plot(x='epoch', y=['loss', 'val_loss'], title="Loss values", ax=ax)
# ax.set_ylim(0, 1000)


# # %%
# f.savefig('losses.png')