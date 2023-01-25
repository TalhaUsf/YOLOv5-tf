#%%
from rich.console import Console
import tensorflow as tf
import numpy as np
import multiprocessing
import os
import tqdm
import xml.etree.ElementTree
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Union

#%%
class IDataloader:
    def __init__(self,
                 data_dir : str,
                 anchors : np.ndarray,
                 ):
        self.data_dir = data_dir
        self.image_size = 640
        self.anchors = anchors
        # self.class_dict will be assigned by the Mixin 

    def write_image_files(self):
        '''
        write the images.txt file listing all the images (without extension) inside the VOC dataset directory
        '''        
        Console().log(f"{self.data_dir} doesnot have images.txt, creating .....")
        # read all the images inside the `data` folder
        images2write = [k.stem for k in Path(self.data_dir, "data").iterdir() if k.suffix == ".jpg"]
        Console().log(f"found {len(images2write)} images in {self.data_dir}/data")
        # write the images to images.txt file
        Path(self.data_dir, 'images.txt').write_text("\n".join(images2write))
        Console().log(f"images.txt created in {self.data_dir}, {len(images2write)} images written")
        return self
    
    def yolo_generate_tf_record(self, class_generate : Union[bool, dict]):
        '''
        🔥 start from here
        spawn processes equal to no. of cpus to write individual tf-record files to the dataset folder
        '''        
        if not os.path.exists(os.path.join(self.data_dir, 'TF')):
            Console().log(f"{self.data_dir} doesnot have TF folder, creating .....")
            os.makedirs(os.path.join(self.data_dir, 'TF'))
        file_names = []
        
        # reding the images
        with open(os.path.join(self.data_dir, 'images.txt'), "r") as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(' ')[0])
        Console().log(f"found {len(file_names)} images in {self.data_dir}/images.txt")
        sentinel = ("", [])
        self.tf_files_to_read = file_names
        # set self.class_dict attribute
        Console().log(f"getting class mapping for dataset")
        
        # # --------------------------------------------------------------------------
        # #                       🔵 making the classes mapping                        
        # # --------------------------------------------------------------------------
        if isinstance(class_generate, bool):
            if class_generate == True:
                # generate the mapping (from training dataloader)
                # internally it will set self.class_dict attribute
                self.get_class_mapping(file_names)
        if isinstance(class_generate, dict):
            # use the mapping provided (for test dataloader)
            self.class_dict = class_generate
        # breakpoint()
        assert hasattr(self, 'class_dict'), "class_dict attribute not found, call `get_class_mapping` method first"
        Console().log(f"➡️\tgot class mapping for dataset\n\n{self.class_dict}")
        
        # # create a shared queue for all the processes
        queue = multiprocessing.Manager().Queue()
        # put all the images into the queue
        for file_name in tqdm.tqdm(file_names, desc='[INFO] preparing TF record', total=len(file_names)):
            queue.put(file_name)
        # then put sentinel into the queue as last element to break the loop
        for _ in range(os.cpu_count()):
            queue.put(sentinel)
        Console().log(f"put all the images into queue for making tf-record files")
        
        process_pool = []
        # start the processes
        Console().log(f"starting {os.cpu_count()} processes to write tf-record files")
        for i in range(os.cpu_count()):
            process = multiprocessing.Process(
                target=self.write_tf_record, args=(queue, sentinel))
            process_pool.append(process)
            process.start()
        for process in process_pool:
            process.join()

    def get_class_mapping(self, file_names : List[str]):
        '''
        takes a list of strings of the xml files and makes a dictionary mapping class names to the integer labels
        calling this method will also assign a class attribute `self.class_dict` to the class object. This must be called 
        for getting the class mapping.
        This must be called ist
        
        
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
        for file_name in tqdm.tqdm(file_names, desc='[class2idx] generating ...', total=len(file_names), colour='green'):        
            path = os.path.join(self.data_dir, "labels", file_name + '.xml')
            root = xml.etree.ElementTree.parse(path).getroot()

            # loop over all the objects in this xml file
            for element in root.iter('object'):
                
                
                class2idx.append(element.find('name').text)
        
        # find unique elements and make a dictionary 
        self.class_dict = {name:idx for idx, name in enumerate(list(set(class2idx)))}
        Console().log(f"class_dict generated ....\n {self.class_dict}", justify='left', highlight=True)    
        

    
    
    
    def write_tf_record(self, queue, sentinel):
        
        def byte_feature(value):
            if not isinstance(value, bytes):
                if not isinstance(value, list):
                    value = value.encode('utf-8')
                else:
                    value = [val.encode('utf-8') for val in value]
            if not isinstance(value, list):
                value = [value]
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        while True:
            file_name = queue.get()

            if file_name == sentinel:
                # when sentinel is found, break the loop becuase that marks the end of queue
                break
            # read jpg image
            
            in_image = self.load_image(file_name)[:, :, ::-1]
            # [xmin, ymin, xmax, ymax] , [class index]
            boxes, label = self.load_label(file_name)

            in_image, boxes = self.resize(in_image, boxes)

            # np.save(f'boxes.npy', boxes)
            # np.save(f'label.npy', label)

            # 🔴 calculate the anchor boxes in the below function call
            y_true_1, y_true_2, y_true_3 = self.process_box(boxes, label)

            in_image = in_image.astype('float32')
            y_true_1 = y_true_1.astype('float32')
            y_true_2 = y_true_2.astype('float32')
            y_true_3 = y_true_3.astype('float32')

            in_image = in_image.tobytes()
            y_true_1 = y_true_1.tobytes()
            y_true_2 = y_true_2.tobytes()
            y_true_3 = y_true_3.tobytes()

            features = tf.train.Features(feature={'in_image': byte_feature(in_image),
                                                'y_true_1': byte_feature(y_true_1),
                                                'y_true_2': byte_feature(y_true_2),
                                                'y_true_3': byte_feature(y_true_3)})
            tf_example = tf.train.Example(features=features)
            opt = tf.io.TFRecordOptions('GZIP')
            with tf.io.TFRecordWriter(os.path.join(self.data_dir, 'TF', file_name + ".tf"), opt) as writer:
                writer.write(tf_example.SerializeToString())

    def load_image(self, file_name):
        '''
        read image using opencv and return the image as np.ndarray

        Parameters
        ----------
        file_name : str
            image name without extension

        Returns
        -------
        image : np.ndarray
            BGR image as np.ndarray 
        '''        
        path = os.path.join(self.data_dir, "data", file_name + '.jpg')
        image = cv2.imread(path)
        return image
    
    def load_label(self, file_name : str):
        '''
        gets annotation filename without extension and loads the associated labels

        Parameters
        ----------
        file_name : str
            xml filename without the extension

        Returns
        -------
        boxes : np.ndarray
            [K, 4] array of boxes
        labels : np.ndarray
            [K, ] array of labels
        
        '''        
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
            labels.append(self.class_dict[element.find('name').text])
        boxes = np.asarray(boxes, np.float32)
        labels = np.asarray(labels, np.int32)
        return boxes, labels
    
    def resize(self, image, boxes=None):
        '''
        resize the given image and boxes to the given image size

        Parameters
        ----------
        image : np.ndarray
            image which is being read
        boxes : np.ndarray, optional
            numpy array of the boxes, by default None

        Returns
        -------
        image_padded : np.ndarray 
        boxes np.ndarray
        '''        
        shape = image.shape[:2]
        scale = min(self.image_size / shape[1], self.image_size / shape[0])
        image = cv2.resize(image, (int(scale * shape[1]), int(scale * shape[0])))

        image_padded = np.zeros(
            [self.image_size, self.image_size, 3], np.uint8)

        dw = (self.image_size - int(scale * shape[1])) // 2
        dh = (self.image_size - int(scale * shape[0])) // 2

        image_padded[dh:int(scale * shape[0]) + dh,
                    dw:int(scale * shape[1]) + dw, :] = image.copy()

        if boxes is None:
            return image_padded, scale, dw, dh

        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh

            return image_padded, boxes




    def process_box(self, boxes, labels):
        anchors_mask = [[6, 7, 8],
                        [3, 4, 5],
                        [0, 1, 2]]
        assert hasattr(self, 'anchors'), "anchors attribute not found"
        anchors = self.anchors
        
        # Console().log(f"➡️\tanchors attribute generated \n\n {self.anchors}")
        # after above line the self.anchors attribute will be available
        
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2  # [Xc, Yc]
        box_size = boxes[:, 2:4] - boxes[:, 0:2]  # [W, H]
        # 🔗 https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
        # y_true_1 corresponds to P5 , image_size // 32
        y_true_1 = np.zeros((self.image_size // 32,
                            self.image_size // 32,
                            3, 5 + len(self.class_dict)), np.float32)
        # y_true_1 corresponds to P4 , image_size // 16
        y_true_2 = np.zeros((self.image_size // 16,
                            self.image_size // 16,
                            3, 5 + len(self.class_dict)), np.float32)
        # y_true_1 corresponds to P3 , image_size // 8
        y_true_3 = np.zeros((self.image_size // 8,
                            self.image_size // 8,
                            3, 5 + len(self.class_dict)), np.float32)

        y_true = [y_true_1, y_true_2, y_true_3]

        box_size = np.expand_dims(box_size, 1)

        min_np = np.maximum(- box_size / 2, - anchors / 2)
        max_np = np.minimum(box_size / 2, anchors / 2)

        whs = max_np - min_np

        overlap = whs[:, :, 0] * whs[:, :, 1]
        union = box_size[:, :, 0] * box_size[:, :, 1] + anchors[:,
                                                                0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10

        iou = overlap / union
        best_match_idx = np.argmax(iou, axis=1)

        ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
        for i, idx in enumerate(best_match_idx):
            feature_map_group = 2 - idx // 3
            ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
            x = int(np.floor(box_centers[i, 0] / ratio))
            y = int(np.floor(box_centers[i, 1] / ratio))
            k = anchors_mask[feature_map_group].index(idx)
            c = labels[i]

            y_true[feature_map_group][y, x, k, :2] = box_centers[i]
            y_true[feature_map_group][y, x, k, 2:4] = box_size[i]
            y_true[feature_map_group][y, x, k, 4] = 1.
            y_true[feature_map_group][y, x, k, 5 + c] = 1.

        return y_true_1, y_true_2, y_true_3 # P5, P4, P3
        





#%%
# # --------------------------------------------------------------------------
# #                              yolo dataloader                        
# # --------------------------------------------------------------------------

class YoloDataloader(IDataloader):
    
    def __init__(self, data_dir : str, anchors : np.ndarray):
        
        super().__init__(data_dir, anchors)
        
        self.description = {'in_image': tf.io.FixedLenFeature([], tf.string),
                            'y_true_1': tf.io.FixedLenFeature([], tf.string),
                            'y_true_2': tf.io.FixedLenFeature([], tf.string),
                            'y_true_3': tf.io.FixedLenFeature([], tf.string)}

    def parse_data(self, tf_record):
        features = tf.io.parse_single_example(tf_record, self.description)

        in_image = tf.io.decode_raw(features['in_image'], tf.float32)
        in_image = tf.reshape(in_image, (self.image_size, self.image_size, 3))
        in_image = in_image / 255.

        y_true_1 = tf.io.decode_raw(features['y_true_1'], tf.float32)
        y_true_1 = tf.reshape(y_true_1,
                              (self.image_size // 32, self.image_size // 32, 3, 5 + len(self.class_dict)))

        y_true_2 = tf.io.decode_raw(features['y_true_2'], tf.float32)
        y_true_2 = tf.reshape(y_true_2,
                              (self.image_size // 16, self.image_size // 16, 3, 5 + len(self.class_dict)))

        y_true_3 = tf.io.decode_raw(features['y_true_3'], tf.float32)
        y_true_3 = tf.reshape(y_true_3,
                              (self.image_size // 8,  self.image_size // 8,  3, 5 + len(self.class_dict)))

        return in_image, y_true_1, y_true_2, y_true_3

    def get_dataloader(self, batch_size : int = 16, epochs : Union[int, None] = None):
        '''
        get the tf.data.Dataset object as dataloader, it will be reading the ground truths from the tfrecord files

        Parameters
        ----------
        batch_size : int, optional
            batch size used to load the samples, by default 16
        epochs : Union[int, None], optional
            if epochs given then dataset will be repeated that much number of times (is only useful in custom training loops), by default None

        Returns
        -------
        _type_
            _description_
        '''        
        file_names = [os.path.join(self.data_dir, 'TF', k+'.tf') for k in tqdm.tqdm(self.tf_files_to_read, desc='reading tfrecord annotation files', colour='magenta', total=len(self.tf_files_to_read))]
        Console().log(f"{len(file_names)} tfrecord files being read for generating ground-truths")
        dataset = tf.data.TFRecordDataset(file_names, 'GZIP')
        # 🔴 following two attributes are for calculating steps/epoch
        self.dataset_length = len(file_names)
        self.get_batch_size = batch_size
        dataset = dataset.map(self.parse_data, os.cpu_count(), tf.data.experimental.AUTOTUNE)
        if epochs is not None:
            dataset = dataset.repeat(epochs + 1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.separate, os.cpu_count(), tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def separate(self, a, b, c, d):
        return (a, (b,c,d))

#%%
# # --------------------------------------------------------------------------
# #                            🔥 anchor generator                        
# # --------------------------------------------------------------------------
class MixinAnchorGenerator:
    '''
    class for generating anchors


    Example
    -------
    >>> gen = MixinAnchorGenerator()._get_info_for_generating_anchors(data_dir="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_train")
    >>> gen.generate_anchor()
    >>> gen.anchors
    array([[ 17.,  21.],
       [ 24.,  51.],
       [ 41., 101.],
       [ 45.,  30.],
       [ 73.,  60.],
       [ 94., 127.],
       [142., 243.],
       [231., 138.],
       [341., 299.]], dtype=float32)
    '''
    # doesnot have an initializer because it must be used as a mixin

    # ☠️this should be called for setting the private attributes of this class
    def _get_info_for_generating_anchors(self,
                                        data_dir: str,
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
        '''        

        self.num_cluster = num_cluster
        self.data_dir = data_dir
        
        
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
        self.anchors = result
        (Path.cwd() / 'anchors.txt').write_text(str(result.tolist()))
        Console().log(f"➡️[Anchor Generator] anchors written to {(Path.cwd() / 'anchors.txt').as_posix()}", justify='left', highlight=True)
        Console().log(f"🔥Anchors : \n{result.tolist()}", style="bold green")
        Console().log(f"🔥Fitness: \n{self.avg_iou(boxes, result)}")

    def get_boxes(self,):
        boxes = []
        file_names = [file_name[:-4]
                    for file_name in os.listdir(os.path.join(self.data_dir, "labels"))]
        
        # 🔴 set the class attribute `self.class_dict` to the class object
        # self.class2idx(file_names)
        for file_name in file_names:
            for box in self.load_label(file_name)[0]:
                boxes.append([box[2] - box[0], box[3] - box[1]])
        # save all the bboxes
        np.save(os.path.join(os.getcwd(), 'boxes.npy'), np.array(boxes))
        return np.array(boxes)

    # def class2idx(self, file_names : List[str]):
    #     '''
    #     takes a list of strings of the xml files and makes a dictionary mapping class names to the integer labels
    #     calling this method will also assign a class attribute `self.class_dict` to the class object. This must be called 
    #     for getting the class mapping.
    #     This must be called ist
        
        
    #     Parameters
    #     ----------
    #     file_names : List[str]
    #         just the name (without extension) of the xml file. for example if the file name is 'a.xml' then just pass 'a' as the argument ['a']

    #     Returns
    #     -------
    #     self
    #     '''
    #     class2idx = []
    #     # loop over all the xml files
    #     for file_name in tqdm(file_names, desc='[class2idx] generating ...', total=len(file_names), colour='green'):        
    #         path = os.path.join(self.data_dir, "labels", file_name + '.xml')
    #         root = xml.etree.ElementTree.parse(path).getroot()

    #         # loop over all the objects in this xml file
    #         for element in root.iter('object'):
                
                
    #             class2idx.append(element.find('name').text)
        
    #     # find unique elements and make a dictionary 
    #     self.class_dict = {name:idx for idx, name in enumerate(list(set(class2idx)))}
    #     Console().log(f"[Anchor Generator] class_dict generated ....\n {self.class_dict}", justify='left', highlight=True)    
        
        
    
    
    def load_label(self, file_name):
        path = os.path.join(self.data_dir, "labels", file_name + '.xml')
        root = xml.etree.ElementTree.parse(path).getroot()

        boxes = []
        # labels = []
        for element in root.iter('object'):
            x_min = float(element.find('bndbox').find('xmin').text)
            y_min = float(element.find('bndbox').find('ymin').text)
            x_max = float(element.find('bndbox').find('xmax').text)
            y_max = float(element.find('bndbox').find('ymax').text)

            boxes.append([x_min, y_min, x_max, y_max])
            # integer labels populating
            # labels.append(self.class_dict[element.find('name').text])
        boxes = np.asarray(boxes, np.float32)
        # labels = np.asarray(labels, np.int32)
        return boxes, None



# %%


if __name__ == "__main__":
    # generate anchors before loading datalaoder
    # because the same dataloader will be used for generating ground truth
    gen = MixinAnchorGenerator()._get_info_for_generating_anchors(data_dir="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_validation")
    gen.generate_anchor()
    anchors = gen.anchors
    
    # generate train / test dataset files
    # a = IDataloader(data_dir="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_validation",
    #                 anchors=anchors).write_image_files()

    # a.yolo_generate_tf_record()
    # class_dict = a.class_dict
    
    a = YoloDataloader(data_dir="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_validation",
                    anchors=anchors).write_image_files()
    a.yolo_generate_tf_record(class_generate=True)
    dataloader = a.get_dataloader()
    for idx, k in enumerate(dataloader):
        breakpoint()
        Console().print(f"BATCH - {idx} loaded")