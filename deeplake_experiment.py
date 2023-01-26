#%%
import deeplake
import os

# %%
dataset_dir = "gopro"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")


# %%
from pathlib import Path

# blur_train = [j for k in Path(train_dir).iterdir() for j in (k/'blur').iterdir()]
# sharp_train = [j for k in Path(train_dir).iterdir() for j in (k/'sharp').iterdir()]

# blur_test = [j for k in Path(test_dir).iterdir() for j in (k/'blur').iterdir()]
# sharp_test = [j for k in Path(test_dir).iterdir() for j in (k/'sharp').iterdir()]
# %%

# # --------------------------------------------------------------------------
# #                    creating custom dataset                        
#           with deeplake 
# # --------------------------------------------------------------------------
# ds = deeplake.empty('GOPRO')

# creating and registering dataset
# ds_val = deeplake.empty('GOPRO_val')
# from tqdm import tqdm

# with ds_val:
#     ds_val.create_tensor('x', htype='image', sample_compression = 'png')
#     ds_val.create_tensor('y', htype='image', sample_compression = 'png')
    
#     for X,Y in tqdm(zip(blur_test, sharp_test), colour='red', desc='VAL-DATASET'):
#         ds_val.append(
#                 {
#                     'x': deeplake.read(X.as_posix()),
#                     'y': deeplake.read(Y.as_posix())
#                 }
#             )

# %%

ds = deeplake.load('GOPRO')
ds_val = deeplake.load('GOPRO_val')
dataloader = ds.tensorflow()
dataloader_val = ds_val.tensorflow()

# %%
import tensorflow as tf

def preprocess(sample):
    image = tf.cast(sample['x'], tf.float32) / 255.0
    label = tf.cast(sample['y'], tf.float32) / 255.0
    image = tf.image.resize(image, [180, 320], preserve_aspect_ratio=True)
    label = tf.image.resize(label, [180, 320], preserve_aspect_ratio=True)
    return image, label

dataloader = dataloader.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(4)
dataloader_val = dataloader_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(4)
# %%
from time import perf_counter

times = []
t0 = perf_counter()
for k in dataloader:
    t1 = perf_counter()
    times.append(t1-t0)
    print(k[0].shape, k[1].shape)
    break
# %%

class psnr_metric(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name='psnr')
        self.data_range = 1.0
    def call(self, y_true, y_pred):        
        


        sub = tf.math.subtract(y_true, y_pred, name='subtract')
        squared = tf.math.square(sub, name='square')
        
        dim = len(squared.shape)
        mse = tf.reduce_mean(squared, list(range(1, dim)))
        psnr = tf.reduce_mean(tf.experimental.numpy.log10(self.data_range**2/mse))
        # psnr = 10 * (self.data_range**2/mse).log10().mean()

        return psnr



#%%

# ==========================================================================
#                          model building                                  
# ==========================================================================
# inferer.build_model(
#     num_rrg=3, num_mrb=2, channels=64,
#     weights_path='low_light_weights_best.h5'
# )


from model_mirnet_v1.mirnet_model import mirnet_model
# from model import MirNetv2
import tensorflow as tf
model = mirnet_model(image_size=None, num_rrg=3, num_mrb=2, channels=64)
# model = MirNetv2()
model.summary()
import tensorflow_model_optimization as tfmot

#%%

quantize_model = tfmot.quantization.keras.quantize_model

# %%

def apply_quantization_to_conv(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

# Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
# to the layers of the model.
annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_conv,
)

del model
q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
# q_aware_model = quantize_model(model)
print(q_aware_model.summary())
del annotated_model


# q_aware_model.compile(optimizer=tf.keras.optimizers.SGD(
#     learning_rate=0.0001, momentum=0.9), loss=tf.keras.losses.MeanSquaredError(name='loss'))

q_aware_model.compile(optimizer=tf.keras.optimizers.SGD(
    learning_rate=1e-6, momentum=0.9), loss=tf.keras.losses.MeanSquaredError(name='loss'), metrics=[psnr_metric()])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.{epoch:02d}-{val_loss:.2f}-{val_psnr:.2f}.hdf5', monitor='val_psnr', verbose=1, save_best_only=False,
    save_weights_only=False, mode='max', save_freq='epoch')
q_aware_model.fit(dataloader, epochs=20, validation_data=dataloader_val, callbacks=[model_ckpt])
# q_aware_model.fit(dataloader, epochs=20, validation_data=dataloader_val, callbacks=[model_ckpt])
# %%



# # ==========================================================================
# #                             prediction                                  
# # ==========================================================================


# loaded_model = tf.keras.models.load_model(
#     filepath="weights.20-0.00.hdf5", custom_objects=None, compile=True, options=None
# )
# print(f'model has been loaded ....')

# _input_layer = tf.keras.Input(
#     shape=[None, None, 3],
#     batch_size=None,
#     name='input_layer'
# )

# _normalized_input = tf.cast(_input_layer, tf.float32) / 255.0
# _resized_input = tf.image.resize(_normalized_input, [180,320])
# _output_layer = loaded_model(_resized_input)
# complete_model = tf.keras.Model(inputs=_input_layer, outputs=_output_layer)


# # %%
# complete_model.save('complete_model')
# # %%
# converter = tf.lite.TFLiteConverter.from_keras_model(complete_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_fp16_model = converter.convert()
# from pathlib import Path
# tflite_model_fp16_file = Path("complete_model.tflite")
# tflite_model_fp16_file.write_bytes(tflite_fp16_model)
# # %%


# # %%
