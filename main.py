import argparse
import multiprocessing
import os
import sys
import numpy as np
import cv2
import numpy
import tensorflow as tf
import tqdm

from nets import nn
from utils import config
from utils import util
from utils.dataset import input_fn, DataLoader
import click
from rich.console import Console
import numpy as np
from construct_model import construct_model, Yolo
from construct_dataloader import MixinAnchorGenerator, YoloDataloader
import atexit




class ComputeLoss(tf.keras.losses.Loss):
    def __init__(self, anchors : np.ndarray, class_dict : dict, output_name : str):
        super().__init__()
        # following two are necessary for multiple devices training and assigning name for this loss function
        self.reduction=tf.keras.losses.Reduction.SUM
        self.name='loss'
        
        self.layer2index = {'P5_20x20':0, 'P4_40x40':1, 'P3_80x80':2}
        self.anchors_index = self.layer2index[output_name]
        self.image_size = 640
        self.anchors = anchors
        self.class_dict = class_dict
    # @staticmethod
    def compute_loss(self, y_pred, y_true, anchors):
        # for a single sample
        grid_size = tf.shape(y_pred)[1:3]
        
        ratio = tf.cast(tf.constant([self.image_size, self.image_size]) / grid_size, tf.float32)
        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
        # breakpoint()
        x_y_offset, pred_boxes, pred_conf, pred_prob = self.process_layer(y_pred, anchors)

        object_mask = y_true[..., 4:5]

        def cond(idx, _):
            return tf.less(idx, tf.cast(batch_size, tf.int32))

        def body(idx, mask):
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4],
                                               tf.cast(object_mask[idx, ..., 0], 'bool'))
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            return idx + 1, mask.write(idx, tf.cast(tf.reduce_max(iou, axis=-1) < 0.2, tf.float32))

        ignore_mask = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        _, ignore_mask = tf.while_loop(cond=cond, body=body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_boxes[..., 0:2] / ratio[::-1] - x_y_offset

        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_boxes[..., 2:4] / anchors
        true_tw_th = tf.where(tf.equal(true_tw_th, 0), tf.ones_like(true_tw_th), true_tw_th)
        pred_tw_th = tf.where(tf.equal(pred_tw_th, 0), tf.ones_like(pred_tw_th), pred_tw_th)
        true_tw_th = tf.math.log(tf.clip_by_value(true_tw_th, 1e-9, 1e+9))
        pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e+9))

        box_loss_scale = y_true[..., 2:3] * y_true[..., 3:4]
        box_loss_scale = 2. - box_loss_scale / tf.cast(self.image_size ** 2, tf.float32)

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale)
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale)

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)

        conf_loss = tf.reduce_sum((conf_loss_pos + conf_loss_neg))

        true_conf = y_true[..., 5:]

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_conf, pred_prob)
        class_loss = tf.reduce_sum(class_loss)

        return xy_loss + wh_loss + conf_loss + class_loss

    def call(self, y_true, y_pred):
        # REFERENCE 🔖 https://github.com/keras-team/keras/blob/9118ea65f40874e915dd1299efd1cc3a7ca2c333/keras/engine/training.py#L816-L848
        loss = 0. # kk = tf.stack([k for k in y_pred], axis=0)
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]] # [P5_20x20, P4_40x40, P3_80x80]
        # breakpoint()
        # 🏹 in custom loop implementation y_true was [p5_true, p4_true, p3_true]
        # 🏹 in custom loop implementation y_pred was [p5_pred, p4_pred, p3_pred]
        # 🔷 Following loop will sum loss against all three grid scales
        # for i in range(len(y_pred)):
        #     loss += self.compute_loss(
        #                                 y_pred=y_pred[i], 
        #                                 y_true=y_true[i], 
        #                                 anchors=anchor_group[i]
        #                             )
        # return loss
        # return self.compute_loss(y_pred=tf.stack([k for k in y_true], axis=0), y_true=tf.stack([k for k in y_pred], axis=0), anchors=anchor_group[0])
        
        return self.compute_loss(y_pred=y_pred, y_true=y_true, anchors=anchor_group[self.anchors_index])
        
    def process_layer(self, feature_map, anchors):
        # gets a single sample
        grid_size = tf.shape(feature_map)[1:3]
        
        ratio = tf.cast(tf.constant([self.image_size, self.image_size]) / grid_size, tf.float32)
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
        
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + len(self.class_dict)])

        box_centers, box_sizes, conf, prob = tf.split(feature_map, [2, 2, 1, len(self.class_dict)], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        box_sizes = box_sizes * ratio[::-1]

        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        return x_y_offset, boxes, conf, prob


    def box_iou(self, pred_boxes, valid_true_boxes):
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        intersect_min = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
        intersect_max = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)

        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)

        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        return intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)








# ==========================================================================
#                             create CLI                                  
# ==========================================================================
@click.group()
def cli():
    pass






def plot_it(step, schedulers, **kwargs):
    '''
    used for plotting the scheduler

    Parameters
    ----------
    step : int
        _description_
    schedulers : _type_
        class of the scheduler
    '''    
    import matplotlib.pyplot as plt
    if not isinstance(schedulers, list):
        schedulers = [schedulers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 3])
    _title = ' '.join([str(k)+'='+str(v)  for k,v in kwargs.items()])
    fig.suptitle(_title, fontsize=16)

    for scheduler in schedulers:
        ax1.plot(range(step), scheduler(range(step)), label=scheduler.name)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.legend()

        ax2.plot(range(step), scheduler(range(step)), label=scheduler.name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend()
    plt.show()
    fig.savefig('scheduler.png')




@cli.command(name='scheduler')
@click.option('--lri', default=1e-2, help='starting lr', type = float, show_default=True)
# @click.option('--decay_steps', default=50, help='number of steps in which to decay', type = int, show_default=True)
@click.option('--alpha', default=1e-2, help='multiplier factor', type = float, show_default=True)
@click.option('--total_steps', default=100, help='total number of steps to show on plot x axis', type = int, show_default=True)
def plot_scheduler(lri, alpha, total_steps):
    init_lr = lri
    # decay_steps = decay_steps
    decay_steps = int(0.8 * total_steps) # after this no. of steps , lr will reach --> init_lr * alpha
    alpha = alpha
    cos_dec1 = tf.keras.experimental.CosineDecay(init_lr, decay_steps, alpha=alpha, name='Cosine Decay')

    # init_lr = 1e-2
    # decay_steps = 50
    # alpha = 0
    # cos_dec2 = tf.keras.experimental.CosineDecay(init_lr, decay_steps, alpha=alpha, name='Cosine Decay 2')

    # init_lr = 1e-3
    # decay_steps = 50
    # alpha = 1e-2
    # cos_dec3 = tf.keras.experimental.CosineDecay(init_lr, decay_steps, alpha=alpha, name='Cosine Decay 3')
    
    plot_it(total_steps, cos_dec1, init_lr=init_lr, decay_steps=decay_steps, alpha=alpha, total_steps=total_steps)



# ==========================================================================
#                             start training                                  
# ==========================================================================
@cli.command(name='train')
@click.option('--model', default='s', help='model type to start training', type = str)
@click.option('--train_dir', help='training dataset dir. pascal voc format', type=str, default="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_train")
@click.option('--test_dir', help='testing dataset dir. pascal voc format', type=str, default="/home/fsuser/AI_ENGINE/yolov5_tf_original/dataset_validation")
def train_model(model, train_dir, test_dir):
    # defining the strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        
        # generate anchors over the training dataset
        gen = MixinAnchorGenerator()._get_info_for_generating_anchors(data_dir=train_dir)
        gen.generate_anchor()
        anchors = gen.anchors
        
        
        # scaling the batch-size
        BATCH_SIZE_PER_REPLICA = 8
        Console().print(f"gpus found {mirrored_strategy.num_replicas_in_sync}", style='green')
        global_batch_size = (BATCH_SIZE_PER_REPLICA *
                            mirrored_strategy.num_replicas_in_sync)
        lr = 0.001 / mirrored_strategy.num_replicas_in_sync
        
        # get dataloader for the training dataset
        train_loader = YoloDataloader(data_dir=train_dir,
                        anchors=anchors).write_image_files()
        train_loader.yolo_generate_tf_record(True)
        class_dict = train_loader.class_dict
        train_DL = train_loader.get_dataloader(batch_size = global_batch_size)
        batch_size = train_loader.get_batch_size
        total_dataset = train_loader.dataset_length
        epochs = 100
        # calculate total training steps
        total_training_steps = (total_dataset // batch_size) * epochs
        
        
        # get dataloader for the testing dataset
        test_loader = YoloDataloader(data_dir=test_dir,
                        anchors=anchors).write_image_files()
        test_loader.yolo_generate_tf_record(class_dict)
        
        test_DL = test_loader.get_dataloader(batch_size = global_batch_size)
        
        
        
        Console().print(f"Class Dict: {class_dict}", style="red on black")
        yolo = construct_model(Yolo, model, class_dict, anchors, True)

        Console().print(yolo.summary(), style='red on green')
        
        # for idx, k in enumerate(test_loader):
        #     Console().print(f"idx -=-> {idx}") # 156 mini-batches were produced
        
        
        
        
        
        # compile model
        
        lr_scheduler = tf.keras.experimental.CosineDecay(0.001, int(0.8*total_training_steps), alpha=0.01, name='Cosine Decay')
        
        yolo.compile(loss={
                         'P5_20x20':ComputeLoss(anchors, class_dict, 'P5_20x20'),
                         'P4_40x40':ComputeLoss(anchors, class_dict, 'P4_40x40'),
                         'P3_80x80':ComputeLoss(anchors, class_dict, 'P3_80x80'),
                    }, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler), run_eagerly=True)
        
        
        
            
        
        yolo.fit(train_DL, epochs=100, validation_data=test_DL, callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir='./logs'), 
                tf.keras.callbacks.CSVLogger(filename='training_logs.csv', separator=',', append=False),
                
                tf.keras.callbacks.ModelCheckpoint(
                                                    filepath='weights.{epoch:03d}-{val_loss:.5f}.hdf5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=False,
                                                    save_weights_only=False,
                                                    mode='min',
                                                    save_freq='epoch')
                ])
        # batch = next(iter(train_loader))
        # breakpoint()
        # predictions = yolo.predict(batch[0])
        
        # print(predictions.shape)
        
    # this line is necessary
    atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore

# numpy.random.seed(12345)
# tf.random.set_seed(12345)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# def train():
#     strategy = tf.distribute.MirroredStrategy()

#     file_names = []
#     with open(os.path.join(config.data_dir, 'train.txt')) as f:
#         for file_name in f.readlines():
#             image_path = os.path.join(config.data_dir, config.image_dir, file_name.rstrip() + '.jpg')
#             label_path = os.path.join(config.data_dir, config.label_dir, file_name.rstrip() + '.xml')
#             if os.path.exists(image_path) and os.path.exists(label_path):
#                 if os.path.exists(os.path.join(config.data_dir, 'TF')):
#                     file_names.append(os.path.join(config.data_dir, 'TF', file_name.rstrip() + '.tf'))
#                 else:
#                     file_names.append(file_name.rstrip())

#     steps = len(file_names) // config.batch_size
#     if os.path.exists(os.path.join(config.data_dir, 'TF')):
#         dataset = DataLoader().input_fn(file_names)
#     else:
#         dataset = input_fn(file_names)
#     breakpoint()
#     dataset = strategy.experimental_distribute_dataset(dataset)

#     with strategy.scope():
#         model = nn.build_model()
#         model.summary()
#         optimizer = tf.keras.optimizers.Adam(nn.CosineLR(steps), 0.937)

#     with strategy.scope():
#         loss_object = nn.ComputeLoss()

#         def compute_loss(y_true, y_pred):
#             # per sample loss (for this replica)
#             total_loss = loss_object(y_pred, y_true)
#             # sum over the batch (for this replica)
#             return tf.reduce_sum(total_loss) / config.batch_size

#     with strategy.scope():
#         def train_step(image, y_true):
#             with tf.GradientTape() as tape:
#                 y_pred = model(image, training=True)
#                 loss = compute_loss(y_true, y_pred)
#             variables = model.trainable_variables
#             gradients = tape.gradient(loss, variables)
#             optimizer.apply_gradients(zip(gradients, variables))
#             return loss

#     with strategy.scope():
#         @tf.function
#         def distributed_train_step(image, y_true):
#             per_replica_losses = strategy.run(train_step, args=(image, y_true))
#             return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

#     def train_fn():
#         if not os.path.exists('weights'):
#             os.makedirs('weights')
#         pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
#         print(f'[INFO] {len(file_names)} data points')
#         for step, inputs in enumerate(dataset):
#             if step % steps == 0:
#                 print(f'Epoch {step // steps + 1}/{config.num_epochs}')
#                 pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
#             step += 1
#             # breakpoint()
#             image, y_true_1, y_true_2, y_true_3 = inputs
#             y_true = (y_true_1, y_true_2, y_true_3)
#             loss = distributed_train_step(image, y_true)
#             pb.add(1, [('loss', loss)])
#             if step % steps == 0:
#                 model.save_weights(os.path.join("weights", f"model_{config.version}.h5"))
#             if step // steps == config.num_epochs:
#                 sys.exit("--- Stop Training ---")

#     train_fn()


# def test():
#     def draw_bbox(image, boxes):
#         for box in boxes:
#             coordinate = numpy.array(box[:4], dtype=numpy.int32)
#             c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
#             cv2.rectangle(image, c1, c2, (255, 0, 0), 1)
#         return image

#     def test_fn():
#         if not os.path.exists('results'):
#             os.makedirs('results')
#         file_names = []
#         with open(os.path.join(config.data_dir, 'test.txt')) as f:
#         # with open(os.path.join(config.data_dir, 'train.txt')) as f:
#             for file_name in f.readlines():
#                 file_names.append(file_name.rstrip())

#         model = nn.build_model(training=False)
#         model.load_weights(f"weights/model_{config.version}.h5", True)

#         for file_name in tqdm.tqdm(file_names):
#             image = cv2.imread(os.path.join(config.data_dir, config.image_dir, file_name + '.jpg'))
#             image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             image_np, scale, dw, dh = util.resize(image_np)
#             image_np = image_np.astype(numpy.float32) / 255.0

#             boxes, scores, labels = model.predict(image_np[numpy.newaxis, ...])

#             boxes, scores, labels = numpy.squeeze(boxes, 0), numpy.squeeze(scores, 0), numpy.squeeze(labels, 0)
#             # breakpoint()
#             boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
#             boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
#             image = draw_bbox(image, boxes)
#             cv2.imwrite(f'results/{file_name}.jpg', image)

#     test_fn()


# def write_tf_record(queue, sentinel):
#     def byte_feature(value):
#         if not isinstance(value, bytes):
#             if not isinstance(value, list):
#                 value = value.encode('utf-8')
#             else:
#                 value = [val.encode('utf-8') for val in value]
#         if not isinstance(value, list):
#             value = [value]
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#     while True:
#         file_name = queue.get()

#         if file_name == sentinel:
#             break
#         # read jpg image
#         # breakpoint()
#         in_image = util.load_image(file_name)[:, :, ::-1]
#         # [xmin, ymin, xmax, ymax] , [class index]
#         boxes, label = util.load_label(file_name)

#         in_image, boxes = util.resize(in_image, boxes)

#         np.save(f'boxes.npy', boxes)
#         np.save(f'label.npy', label)
        
        
#         y_true_1, y_true_2, y_true_3 = util.process_box(boxes, label)

#         in_image = in_image.astype('float32')
#         y_true_1 = y_true_1.astype('float32')
#         y_true_2 = y_true_2.astype('float32')
#         y_true_3 = y_true_3.astype('float32')

#         in_image = in_image.tobytes()
#         y_true_1 = y_true_1.tobytes()
#         y_true_2 = y_true_2.tobytes()
#         y_true_3 = y_true_3.tobytes()

#         features = tf.train.Features(feature={'in_image': byte_feature(in_image),
#                                               'y_true_1': byte_feature(y_true_1),
#                                               'y_true_2': byte_feature(y_true_2),
#                                               'y_true_3': byte_feature(y_true_3)})
#         tf_example = tf.train.Example(features=features)
#         opt = tf.io.TFRecordOptions('GZIP')
#         with tf.io.TFRecordWriter(os.path.join(config.data_dir, 'TF', file_name + ".tf"), opt) as writer:
#             writer.write(tf_example.SerializeToString())


# def generate_tf_record():
#     if not os.path.exists(os.path.join(config.data_dir, 'TF')):
#         os.makedirs(os.path.join(config.data_dir, 'TF'))
#     file_names = []
#     # with open(os.path.join(config.data_dir, 'train.txt')) as reader:
#     with open(os.path.join(config.data_dir, 'test.txt')) as reader:
#         for line in reader.readlines():
#             file_names.append(line.rstrip().split(' ')[0])
#     sentinel = ("", [])
#     # create a shared queue for all the processes
#     queue = multiprocessing.Manager().Queue()
#     # put all the images into the queue
#     for file_name in tqdm.tqdm(file_names, desc='[INFO] preparing TF record', total=len(file_names)):
#         queue.put(file_name)
#     # then put sentinel into the queue as last element to break the loop
#     for _ in range(os.cpu_count()):
#         queue.put(sentinel)
#     print('[INFO] generating TF record')
#     process_pool = []
#     # start the processes
#     for i in range(os.cpu_count()):
#         process = multiprocessing.Process(target=write_tf_record, args=(queue, sentinel))
#         process_pool.append(process)
#         process.start()
#     for process in process_pool:
#         process.join()



# class AnchorGenerator:
#     def __init__(self, num_cluster):
#         self.num_cluster = num_cluster
    
#     def iou(self, boxes, clusters):  # 1 box -> k clusters
        
#         n = boxes.shape[0]
#         k = self.num_cluster
#         # breakpoint()
#         box_area = boxes[:, 0] * boxes[:, 1]
#         box_area = box_area.repeat(k)
#         box_area = numpy.reshape(box_area, (n, k))

#         cluster_area = clusters[:, 0] * clusters[:, 1]
#         cluster_area = numpy.tile(cluster_area, [1, n])
#         cluster_area = numpy.reshape(cluster_area, (n, k))

#         box_w_matrix = numpy.reshape(boxes[:, 0].repeat(k), (n, k))
#         cluster_w_matrix = numpy.reshape(numpy.tile(clusters[:, 0], (1, n)), (n, k))
#         min_w_matrix = numpy.minimum(cluster_w_matrix, box_w_matrix)

#         box_h_matrix = numpy.reshape(boxes[:, 1].repeat(k), (n, k))
#         cluster_h_matrix = numpy.reshape(numpy.tile(clusters[:, 1], (1, n)), (n, k))
#         min_h_matrix = numpy.minimum(cluster_h_matrix, box_h_matrix)
#         inter_area = numpy.multiply(min_w_matrix, min_h_matrix)

#         return inter_area / (box_area + cluster_area - inter_area)
    
#     def avg_iou(self, boxes, clusters):
#         accuracy = numpy.mean([numpy.max(self.iou(boxes, clusters), axis=1)])
#         return accuracy
    
#     def generator(self, boxes, k, dist=numpy.median):
#         box_number = boxes.shape[0]
#         last_nearest = numpy.zeros((box_number,))
#         # select randomly 9 boxes and assume as clusters
#         clusters = boxes[numpy.random.choice(box_number, k, replace=False)]  # init k clusters
#         while True:
#             distances = 1 - self.iou(boxes, clusters)

#             current_nearest = numpy.argmin(distances, axis=1)
#             if (last_nearest == current_nearest).all():
#                 break  # clusters won't change
#             for cluster in range(k):
#                 clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
#             last_nearest = current_nearest

#         return clusters
    
#     def generate_anchor(self):
#         boxes = self.get_boxes()
#         result = self.generator(boxes, k=self.num_cluster)
#         # breakpoint()
#         result = result[numpy.lexsort(result.T[0, None])]
#         from pathlib import Path
#         # breakpoint()
#         (Path.cwd() /'anchors.txt').write_text(str(result.tolist()))
#         print("\nAnchors: \n{}".format(result))
#         print("\nFitness: {:.4f}".format(self.avg_iou(boxes, result)))
    
#     @staticmethod
    
#     def get_boxes():
#         boxes = []
#         file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.label_dir))]
#         for file_name in file_names:
#             for box in util.load_label(file_name)[0]:
#                 boxes.append([box[2] - box[0], box[3] - box[1]])
#         # save all the bboxes
#         numpy.save(os.path.join(os.getcwd(), 'boxes.npy'), numpy.array(boxes))
#         return numpy.array(boxes)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--anchor', action='store_true')
#     parser.add_argument('--record', action='store_true')
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--test', action='store_true')

#     args = parser.parse_args()
#     if args.anchor:
#         AnchorGenerator(9).generate_anchor()
#     if args.record:
#         generate_tf_record()
#     if args.train:
#         train()
#     if args.test:
#         test()



if __name__ == '__main__':
    cli()