import tqdm
import cv2
import tarfile
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

ROOT_DIR = 'D:\\datasets\\'

def generic_processing(temp_ds, test_ds, info, batch_size, transforms):
    temp_ds = temp_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    eval_ds = temp_ds.take(int(info.splits['train'].num_examples / 10.))
    train_ds = temp_ds.skip(int(info.splits['train'].num_examples / 10.))
    
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(info.splits['train'].num_examples)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    eval_ds = eval_ds.batch(batch_size)
    eval_ds = eval_ds.cache()
    eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return (train_ds, eval_ds, test_ds)

def load_mnist(batch_size):
    (temp_ds, test_ds), info = tfds.load('mnist',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        return img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_noisy_mnist(batch_size):
    (temp_ds, test_ds), info = tfds.load('mnist',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        noisy_img = img + .25 * tf.random.normal(shape=img.shape)
        noisy_img = tf.clip_by_value(noisy_img, clip_value_min=0., clip_value_max=1.)
        return noisy_img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_fashion_mnist(batch_size):
    (temp_ds, test_ds), info = tfds.load('fashion_mnist',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        return img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_noisy_fashion_mnist(batch_size):
    (temp_ds, test_ds), info = tfds.load('fashion_mnist',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        noisy_img = img + .25 * tf.random.normal(shape=img.shape)
        noisy_img = tf.clip_by_value(noisy_img, clip_value_min=0., clip_value_max=1.)
        return noisy_img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_cifar(batch_size):
    (temp_ds, test_ds), info = tfds.load('cifar10',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        return img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_noisy_cifar(batch_size):
    (temp_ds, test_ds), info = tfds.load('cifar10',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=ROOT_DIR)
    
    def transforms(image, label):
        img = tf.cast(image, tf.float32) / 255.
        noisy_img = img + .15 * tf.random.normal(shape=img.shape)
        noisy_img = tf.clip_by_value(noisy_img, clip_value_min=0., clip_value_max=1.)
        return noisy_img, img
    
    return generic_processing(temp_ds, test_ds, info, batch_size, transforms)


def load_lfw(batch_size):
    def decode_image_from_raw_bytes(raw_bytes):
        img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def transforms(x, y):
        img = tf.cast(x, tf.float32) / 255.
        return img, img
    crop = 30
    size = 128
    lfw_path = f'{ROOT_DIR}lfw.tgz'
    all_photos = []
    ds_size = 0
    with tarfile.open(lfw_path) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            if m.isfile() and m.name.endswith('.jpg'):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[crop:-crop, crop:-crop]
                img = cv2.resize(img, (size, size))
                all_photos.append(img)
                ds_size += 1
    all_photos = np.stack(all_photos)
    temp_ds = tf.data.Dataset.from_tensor_slices((all_photos, all_photos))
    temp_ds = temp_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    temp_ds = temp_ds.shuffle(ds_size, seed=42)
    
    test_ds = temp_ds.take(int(ds_size / 10.))
    train_ds = temp_ds.skip(int(ds_size / 10.))
    eval_ds = train_ds.take(int(ds_size / 10.))
    train_ds = train_ds.skip(int(ds_size / 10.))
    
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(int(ds_size / 2.))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    eval_ds = eval_ds.batch(batch_size)
    eval_ds = eval_ds.cache()
    eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return (train_ds, eval_ds, test_ds)


def load_blurry_lfw(batch_size):
    def decode_image_from_raw_bytes(raw_bytes):
        img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def transforms(x, y):
        return tf.cast(x, tf.float32) / 255., tf.cast(y, tf.float32) / 255.
    crop = 30
    size = 128
    blur_k = 6
    lfw_path = f'{ROOT_DIR}lfw.tgz'
    all_photos = []
    all_blured_photos = []
    ds_size = 0
    with tarfile.open(lfw_path) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            if m.isfile() and m.name.endswith('.jpg'):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[crop:-crop, crop:-crop]
                img = cv2.resize(img, (size, size))
                all_photos.append(img)
                all_blured_photos.append(cv2.blur(img, (blur_k, blur_k), cv2.BORDER_DEFAULT))
                ds_size += 1
    all_photos = np.stack(all_photos)
    all_blured_photos = np.stack(all_blured_photos)
    temp_ds = tf.data.Dataset.from_tensor_slices((all_blured_photos, all_photos))
    temp_ds = temp_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    temp_ds = temp_ds.shuffle(ds_size, seed=42)
    
    test_ds = temp_ds.take(int(ds_size / 10.))
    train_ds = temp_ds.skip(int(ds_size / 10.))
    eval_ds = train_ds.take(int(ds_size / 10.))
    train_ds = train_ds.skip(int(ds_size / 10.))
    
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(int(ds_size / 2.))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    eval_ds = eval_ds.batch(batch_size)
    eval_ds = eval_ds.cache()
    eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return (train_ds, eval_ds, test_ds)