import os
import numpy

width = [0.50, 0.75, 1.0, 1.25]
depth = [0.33, 0.67, 1.0, 1.33]

versions = ['s', 'm', 'l', 'x']
# data_dir = os.path.join('..', 'Dataset', 'COCO')
data_dir = os.path.join('..', 'dataset_train')
# data_dir = os.path.join('..', 'dataset_validation')

# threshold = 0.3
threshold = 0.9
max_boxes = 150
# image_dir = 'images'
image_dir = 'data'
label_dir = 'labels'

num_epochs = 300
batch_size = 32
image_size = 640
# class_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
            #   'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
            #   'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
            #   'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
            #   'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
            #   'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39,
            #   'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47,
            #   'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54,
            #   'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61,
            #   'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68,
            #   'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75,
            #   'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}

class_dict = {                                                                                                                        
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
version = 'm'
# anchors = numpy.array([[8., 9.],    [16., 24.],   [28., 58.],
#                        [41., 25.],  [58., 125.],  [71., 52.],
#                        [129., 97.], [163., 218.], [384., 347.]], numpy.float32)

anchors = numpy.array([[17.0, 21.0], [24.0, 51.0], [41.0, 100.0], 
                       [45.0, 31.0], [75.0, 61.0], [94.0, 129.0], 
                       [143.0, 245.0], [232.0, 138.0], [342.0, 299.0]], numpy.float32)