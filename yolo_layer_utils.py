import tensorflow as tf
from tensorflow.keras import backend
import numpy as np
from rich.console import Console


# prediction layer needed by all the variants of model
class Predict(tf.keras.layers.Layer):
    def __init__(self, anchors : np.ndarray, image_size : int, n_classes : int, max_boxes : int):
        super().__init__()
        self.anchors = anchors
        self.image_size = image_size
        self.n_classes = n_classes
        self.max_boxes = max_boxes
        Console().log(f"anchors: {anchors} image_size: {image_size} n_classes: {n_classes} max_boxes: {max_boxes}", style="bold red")
        
        
    def call(self, inputs, **kwargs):
        
        # group spatial prediction tensor w.r.t anchor sizes
        # [P5, P4, P3] = inputs
        y_pred = [(inputs[0], self.anchors[6:9]), # (20, 20) # P5
                  (inputs[1], self.anchors[3:6]), # (40, 40) # P4
                  (inputs[2], self.anchors[0:3])] # (80, 80) # P3

        boxes_list, conf_list, prob_list = [], [], []
        for result in [Predict.process_layer(feature_map=feature_map, anchors=anchors, image_size=self.image_size, n_classes=self.n_classes) for (feature_map, anchors) in y_pred]:
            x_y_offset, box, conf, prob = result
            grid_size = tf.shape(x_y_offset)[:2]
            box = tf.reshape(box, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf = tf.reshape(
                conf, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob = tf.reshape(
                prob, [-1, grid_size[0] * grid_size[1] * 3, self.n_classes])
            boxes_list.append(box)
            conf_list.append(tf.sigmoid(conf))
            prob_list.append(tf.sigmoid(prob))

        boxes = tf.concat(boxes_list, axis=1)
        conf = tf.concat(conf_list, axis=1)
        prob = tf.concat(prob_list, axis=1)

        center_x, center_y, w, h = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        x_max = center_x + w / 2
        y_max = center_y + h / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        outputs = tf.map_fn(fn=Predict.compute_nms,
                            elems=[boxes, conf * prob], # threshold = 0.9, max_boxes : int = 150):
                            dtype=['float32', 'float32', 'int32'],
                            parallel_iterations=100)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.max_boxes, 4),
                (input_shape[1][0], self.max_boxes),
                (input_shape[1][0], self.max_boxes), ]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs) + 1) * [None]

    def get_config(self):
        return super().get_config()
    
    
    # helper function
    @staticmethod
    def process_layer(feature_map, anchors, image_size : int = 640, n_classes : int = 20):
    
        
        grid_size = tf.shape(feature_map)[1:3]
        ratio = tf.cast(tf.constant(
            [image_size, image_size]) / grid_size, tf.float32)
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0])
                            for anchor in anchors]

        feature_map = tf.reshape(
            feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + n_classes])

        box_centers, box_sizes, conf, prob = tf.split(
            feature_map, [2, 2, 1, n_classes], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.cast(tf.reshape(
            x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        box_sizes = box_sizes * ratio[::-1]

        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        return x_y_offset, boxes, conf, prob

    @staticmethod
    def compute_nms(args, threshold : float = 0.9, max_boxes : int = 150):
        
        boxes, classification = args
        
        def nms_fn(score, label):
            score_indices = tf.where(backend.greater(score, threshold))

            filtered_boxes = tf.gather_nd(boxes, score_indices)
            filtered_scores = backend.gather(score, score_indices)[:, 0]

            nms_indices = tf.image.non_max_suppression(
                filtered_boxes, filtered_scores, max_boxes, 0.1)
            score_indices = backend.gather(score_indices, nms_indices)

            label = tf.gather_nd(label, score_indices)
            score_indices = backend.stack([score_indices[:, 0], label], axis=1)

            return score_indices

        all_indices = []
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((backend.shape(scores)[0],), dtype='int64')
            all_indices.append(nms_fn(scores, labels))
        indices = backend.concatenate(all_indices, axis=0)

        scores = tf.gather_nd(classification, indices)
        labels = indices[:, 1]
        scores, top_indices = tf.nn.top_k(scores, k=backend.minimum(
            max_boxes, backend.shape(scores)[0]))

        indices = backend.gather(indices[:, 0], top_indices)
        boxes = backend.gather(boxes, indices)
        labels = backend.gather(labels, top_indices)

        pad_size = backend.maximum(0, max_boxes - backend.shape(scores)[0])

        boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
        scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
        labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
        labels = backend.cast(labels, 'int32')

        boxes.set_shape([max_boxes, 4])
        scores.set_shape([max_boxes])
        labels.set_shape([max_boxes])

        return [boxes, scores, labels]


prediction_layer = Predict(
                np.array([[17.0, 21.0], [24.0, 51.0], [41.0, 100.0], 
                       [45.0, 31.0], [75.0, 61.0], [94.0, 129.0], 
                       [143.0, 245.0], [232.0, 138.0], [342.0, 299.0]], np.float32),
                640,
                20,
                150
                )

pred1 = tf.random.normal(
    shape=[1, 20, 20, 3, 25], mean=0.0, stddev=1.0, dtype=tf.float32)
pred2 = tf.random.normal(
    shape=[1, 40, 40, 3, 25], mean=0.0, stddev=1.0, dtype=tf.float32)
pred3 = tf.random.normal(
    shape=[1, 80, 80, 3, 25], mean=0.0, stddev=1.0, dtype=tf.float32)

pred = [pred1, pred2, pred3]
out = prediction_layer(pred)
Console().log(out)