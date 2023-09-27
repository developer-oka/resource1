import tensorflow as tf
import os
import cv2
import numpy as np

pic_dir = ""
size = None

saved_model_dir = "/work/data/pet/saved_model_tflite_out/saved_model"
filename = "/work/data/pet//work/data/pet/pet.tflite"
pic_dir = "/work/data/pet/images/"
size = (640, 640)


def representative_dataset():
    filenames = os.listdir(pic_dir)
    cnt = 0
    _dict = []
    for file in filenames:
        cnt = cnt + 1
        if cnt > 300:
            break

        img = cv2.imread(pic_dir + file)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_bgr, size)
        img_normalized = img_resize / 128 - 1
        img_np_expanded = np.expand_dims(img_normalized, axis=0)
        inputs = img_np_expanded.astype(np.float32)
        yield [inputs]


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
open(filename, "wb").write(tflite_quant_model)

