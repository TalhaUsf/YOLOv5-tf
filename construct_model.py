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


class MixinAnchorGenerator:
    # doesnot have an initializer because it must be used as a mixin

    # ☠️this should be called for setting the private attributes of this class
    def _get_info_for_generating_anchors(self,
                                         data_dir: str,
                                         class_dict: dict,
                                         num_cluster: int = 9):
        '''
        This method sets attributes of this mixin class that are required for generating anchors sub-sequently

        Parameters
        ----------
        num_cluster : int , optional
            Number of clusters , 9 by default. There are 3 scales of prediction with each scale associated
            with 3 anchor boxes this makes it a total of 9 anchor boxes
        data_dir : str
            directory which contains dataset in PascalVOC format, it must have `data` and `labels` directories
        class_dict : dict
            dictionary with keys as class names and values as class ids integers
        '''        

        self.num_cluster = num_cluster
        self.data_dir = data_dir
        self.class_dict = class_dict
        
        return self

    def iou(self, boxes, clusters):  # 1 box -> k clusters

        n = boxes.shape[0]
        k = self.num_cluster
        # breakpoint()
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        return inter_area / (box_area + cluster_area - inter_area)

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def generator(self, boxes, k, dist=np.median):
    
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        # select randomly 9 boxes and assume as clusters
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(
                    boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def generate_anchor(self):
        '''
        🟦 This is the method that needs to be called for starting the calculation of anchor boxes
        
        Examples
        --------
        If this class is to be used as standalone class then it can be used as follows: 

        >>> MixinAnchorGenerator()._get_info_for_generating_anchors(data_dir = 'a/b/c',
                                         class_dict = {'a' : 0, 'b' : 1},
                                         num_cluster = 9).generate_anchor()
        [17:40:32] ➡️[Anchor Generator] boxes generated
        [17:40:32] ➡️[Anchor Generator] Clusters calculated  
        
        '''        
        boxes = self.get_boxes()
        Console().log(f"➡️[Anchor Generator] boxes generated", justify='left', highlight=True)
        result = self.generator(boxes, k=self.num_cluster)
        Console().log(f"➡️[Anchor Generator] Clusters calculated", justify='left', highlight=True)
        # breakpoint()
        result = result[np.lexsort(result.T[0, None])]
        from pathlib import Path
        # breakpoint()
        (Path.cwd() / 'anchors.txt').write_text(str(result.tolist()))
        Console().log(f"➡️[Anchor Generator] anchors written to {(Path.cwd() / 'anchors.txt').as_posix()}", justify='left', highlight=True)
        Console().log(f"🔥Anchors : \n{result.tolist()}", style="bold green")
        Console().log(f"🔥Fitness: \n{self.avg_iou(boxes, result)}")

    def get_boxes(self,):
        boxes = []
        file_names = [file_name[:-4]
                      for file_name in os.listdir(os.path.join(self.data_dir, "labels"))]
        for file_name in file_names:
            for box in self.load_label(file_name)[0]:
                boxes.append([box[2] - box[0], box[3] - box[1]])
        # save all the bboxes
        np.save(os.path.join(os.getcwd(), 'boxes.npy'), np.array(boxes))
        return np.array(boxes)

    def class2idx(self, file_names : List[str]):
        '''
        takes a list of strings of the xml files and makes a dictionary mapping class names to the integer labels
        calling this method will also assign a class attribute `self.class_dict` to the class object. This must be called 
        for getting the class mapping.
        
        
        
        Parameters
        ----------
        file_names : List[str]
            just the name (without extension) of the xml file. for example if the file name is 'a.xml' then just pass 'a' as the argument ['a']

        Returns
        -------
        self
        '''
        class2idx = []
        # loop over all the xml files
        for file_name in tqdm(file_names, desc='[class2idx] generating ...', total=len(file_names), colour='green'):        
            path = os.path.join(self.data_dir, "labels", file_name + '.xml')
            root = xml.etree.ElementTree.parse(path).getroot()

            # loop over all the objects in this xml file
            for element in root.iter('object'):
                
                
                class2idx.append(element.find('name').text)
        
        # find unique elements and make a dictionary 
        self.class_dict = {name:idx for idx, name in enumerate(list(set(class2idx)))}
        Console().log(f"[Anchor Generator] class_dict generated ....\n {self.class_dict}", justify='left', highlight=True)    
        
        return self
    
    
    def load_label(self, file_name):
        path = os.path.join(self.data_dir, "labels", file_name + '.xml')
        root = xml.etree.ElementTree.parse(path).getroot()

        boxes = []
        labels = []
        for element in root.iter('object'):
            x_min = float(element.find('bndbox').find('xmin').text)
            y_min = float(element.find('bndbox').find('ymin').text)
            x_max = float(element.find('bndbox').find('xmax').text)
            y_max = float(element.find('bndbox').find('ymax').text)

            boxes.append([x_min, y_min, x_max, y_max])
            # integer labels populating
            labels.append(self.class_dict[element.find('name').text])
        boxes = np.asarray(boxes, np.float32)
        labels = np.asarray(labels, np.int32)
        return boxes, labels


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
        # anchor boxes computed over the dataset
        self.anchors = np.array([[17.0, 21.0], [24.0, 51.0], [41.0, 100.0],
                                 [45.0, 31.0], [75.0, 61.0], [94.0, 129.0],
                                 [143.0, 245.0], [232.0, 138.0], [342.0, 299.0]], np.float32)
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
class Yolo(MixinAnchorGenerator, IYoloFamily): # mixin should be inherited ist because constructor of IYoloFamily needs to be used by Yolo

    # override the method and change the arguments
    def build_model(self, version: str, n_classes: int, training: bool):
        self.class_dict = n_classes
        self.version_selected = version
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
        p3 = tf.keras.layers.Conv2D(3 * (self.class_dict + 5), 1, name=f'p3_{self.image_size//8}x{self.image_size//8}x3x{self.class_dict+5}',
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        x = self.conv(x, int(round(width * 256)), 3, 2)
        x = tf.keras.layers.concatenate([x, x4])
        x = self.csp(x, int(round(width * 512)), int(round(depth * 3)), False)
        # ⚠️ P4 --> image_size // 16
        p4 = tf.keras.layers.Conv2D(3 * (self.class_dict + 5), 1, name=f'p4_{self.image_size//16}x{self.image_size//16}x3x{self.class_dict+5}',
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        x = self.conv(x, int(round(width * 512)), 3, 2)
        x = tf.keras.layers.concatenate([x, x3])
        x = self.csp(x, int(round(width * 1024)), int(round(depth * 3)), False)
        # ⚠️ P5 --> self.image_size // 32
        p5 = tf.keras.layers.Conv2D(3 * (self.class_dict + 5), 1, name=f'p5_{self.image_size//32}x{self.image_size//32}x3x{self.class_dict+5}',
                                    kernel_initializer=super().initializer, kernel_regularizer=super().l2)(x)

        # ==========================================================================
        #                           ☠️  Model Output layer Surgery
        # ==========================================================================
        if training:
            # ⚡ in training model the prediction will be performed over 3 feature maps
            return tf.keras.Model(inputs, [p5, p4, p3])
        else:
            # ⚡ in inference model the NMS layer will be made a part of the model, the last prediction layer will hold no learnable parameters
            return tf.keras.Model(inputs, Predict(anchors=self.anchors, image_size=self.image_size, n_classes=self.class_dict, max_boxes=self.max_boxes)([p5, p4, p3]))


# common function to construct the model

def construct_model(cls, version: str, n_classes: int, is_training: bool) -> IYoloFamily:
    '''
    this function will be used to build all variants of the yolo family of models

    Parameters
    ----------
    version : str
        can be one of the following: 's', 'm', 'l', 'x'
    n_classes : int
        number of classes in the dataset
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
    model_class._get_info_for_generating_anchors(data_dir="../dataset_validation/",
                                         class_dict= {                                                                                                                        
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
                                                                    }
                                         ).generate_anchor()
    model = model_class.build_model(
        version=version, n_classes=n_classes, training=is_training)

    return model


def main():

    yolo = construct_model(Yolo, 's', 20, True)

    print(yolo.summary())

# %%


if __name__ == '__main__':
    main()
