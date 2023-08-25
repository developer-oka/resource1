import random

import hashlib
import io
import os
import shutil

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# ファイルを入れるディレクトリのパス
output_path_train = "/work/data/train_face.tfrecord"
output_path_val = "/work/data/val_face.tfrecord"

path_ano_train = "/mnt/hgfs/D/share/face/wider_face_split/wider_face_train_bbx_gt.txt"
path_ano_val = "/mnt/hgfs/D/share/face/wider_face_split/wider_face_val_bbx_gt.txt"

label_map_path = "/work/data/label_map_face.pbtxt"

# トレーニング用
dir_path_pic_train = "/mnt/hgfs/D/share/face/WIDER_train/images/"
# 評価用
dir_path_pic_val = "/mnt/hgfs/D/share/face/WIDER_val/images/"

def dict_to_tf_example(data,
                       label_map_dict,
                       full_path,
                       ignore_difficult_instances=False):
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    writer = tf.python_io.TFRecordWriter(output_path_train)

    annotations = open('./wider_face_split/wider_face_train_bbx_gt.txt')
    lines = annotations.readlines()
    line_num = len(lines)

    i = 0
    j = 0

    while i < line_num:
        imgName = str(lines[i].rstrip('\n'))
        i = i + 1
        face_num = lines[i]
        i = i + 1
        i = i + int(face_num)

    for filename in dir_path_pic_train:
        pic_path = os.path.join(dir_path_src, filename)
        ano_path = os.path.join(dir_path_src_xml, os.path.splitext(filename)[0] + ".xml")
        print(pic_path)
        print(ano_path)
        with tf.gfile.GFile(ano_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict, pic_path)
        writer.write(tf_example.SerializeToString())

    writer.close()

    writer = tf.python_io.TFRecordWriter(output_path_val)

    for filename in file_list_val:
        pic_path = os.path.join(dir_path_src, filename)
        ano_path = os.path.join(dir_path_src_xml, os.path.splitext(filename)[0] + ".xml")
        print(pic_path)
        print(ano_path)
        with tf.gfile.GFile(ano_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict, pic_path)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
