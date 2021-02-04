import tqdm
import cv2
import tarfile
import numpy as np
import os
import glob
from PIL import Image
import shutil
import random

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

def load_bsds500(batch_size):
    def preprocess_bsds500(tar_filename):
        """
        Complete pre-process of the whole dataset and saving in a tar file.
        Not optimized for easier understanding (run only once anyway).
        """
        def create_subimages(img, image_path, destination_dir, patch_width=160, patch_height=160):
            """
            Sub-images are randomly rotated
            """
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img_width, img_height = img.size
            img_height -= img_height % patch_height
            img_width -= img_width % patch_width
            k = 1
            for i in range(0, img_height, patch_height):
                for j in range(0, img_width, patch_width):
                    box = (j, i, j + patch_width, i + patch_height)
                    patch = Image.new('RGB', (patch_height, patch_width), 255)
                    patch.paste(img.crop(box))
                    patch = patch.rotate(random.randint(0, 3) * 90)
                    patch.save(f'{destination_dir}{image_name}_{k}.jpg')
                    k += 1

        def copy_image(img, image_path, destination_dir):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img.save(f'{destination_dir}{image_name}.jpg')

        def create_compressed(img, destination_path, level):
            img.save(destination_path, format='jpeg', quality=level)

        def create_noisified(img, destination_path, level):
            img = np.array(img).astype(np.float32)
            img /= 255.
            img += level * np.random.normal(size=img.shape)
            img = np.clip(img, 0., 1.)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(destination_path)

        def create_downscaled(img, destination_path, factor):
            img_width, img_height = img.size
            img = img.resize((int(img_width / factor), int(img_height / factor)), Image.BICUBIC)
            img = img.resize((img_width, img_height), Image.BICUBIC)
            img.save(destination_path)

        print('BSDS500 preprocessing starts, this can take some minutes')
        dataset_dir = f'{ROOT_DIR}bsds500\\'
        temp_dir = f'{ROOT_DIR}temp\\'
        # Delete and create temp dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Write sub-images in temp/targets/train dir and copy others in corresponding temp/dirs
        for local_dir in ['train', 'test', 'val']:
            destination_dir = f'{temp_dir}targets\\{local_dir}\\'
            os.makedirs(destination_dir)
            igms_paths = glob.glob(f'{dataset_dir}{local_dir}\\*.jpg')
            for image_path in igms_paths:
                # Rotate images so that they are all of the same dimensions
                img = Image.open(image_path)
                img_width, img_height = img.size
                box = (0, 0, img_width - 1, img_height - 1)
                new_img = Image.new('RGB', (img_width - 1, img_height - 1), 255)
                new_img.paste(img.crop(box))
                if img_height > img_width:
                    new_img = new_img.rotate(90, expand=True)
                # Create sub-images or copy the image
                if local_dir == 'train':
                    create_subimages(new_img, image_path, destination_dir)
                else:
                    copy_image(new_img, image_path, destination_dir)
        # Preprocess sub-images, create a downscaled, noisified and compressed version for each train and val image
        for local_dir in ['train', 'val']:
            destination_dir = f'{temp_dir}data\\{local_dir}\\'
            os.makedirs(destination_dir)
            igms_paths = glob.glob(f'{temp_dir}targets\\{local_dir}\\*.jpg')
            for image_path in igms_paths:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                img = Image.open(image_path)
                create_noisified(img, f'{destination_dir}{image_name}_n.jpg', [.15, .25, .50][random.randint(0, 2)])
                create_downscaled(img, f'{destination_dir}{image_name}_d.jpg', random.randint(2, 4))
                create_compressed(img, f'{destination_dir}{image_name}_c.jpg', random.randint(1, 4) * 10)
        # Preprocess all test sets
        igms_paths = glob.glob(f'{temp_dir}targets\\test\\*.jpg')
        for image_path in igms_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            img = Image.open(image_path)
            for local_dir in ['noise_15', 'noise_25', 'noise_50',
                          'upscale_2', 'upscale_3', 'upscale_4',
                          'compress_10', 'compress_20', 'compress_30', 'compress_40']:
                destination_dir = f'{temp_dir}data\\test\\{local_dir}\\'
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                action, level = local_dir.split('_')
                if action == 'noise':
                    create_noisified(img, f'{destination_dir}{image_name}.jpg', int(level) / 100.)
                if action == 'upscale':
                    create_downscaled(img, f'{destination_dir}{image_name}.jpg', int(level))
                if action == 'compress':
                    create_compressed(img, f'{destination_dir}{image_name}.jpg', int(level))
        # Create tar file with temp dir
        with tarfile.open(tar_filename, "w:gz") as tar:
            tar.add(temp_dir, arcname=os.path.basename(temp_dir))
        # Delete temp dir
        shutil.rmtree(temp_dir)
        print('Preprocessing over')
    def decode_image_from_raw_bytes(raw_bytes):
        img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def transforms(x, y):
        return tf.cast(x, tf.float32) / 255., tf.cast(y, tf.float32) / 255.
    
    tar_filename = f'{ROOT_DIR}bsds500.tgz'
    # If tar file does not exists
    if not os.path.exists(tar_filename):
        preprocess_bsds500(tar_filename)
    # Create tf datasets from tar file
    train_data = []
    val_data = []
    test_data = {
        'compress_10': [],
        'compress_20': [],
        'compress_30': [],
        'compress_40': [],
        'noise_15': [],
        'noise_25': [],
        'noise_50': [],
        'upscale_2': [],
        'upscale_3': [],
        'upscale_4': [],
    }
    train_targets = []
    val_targets = []
    test_targets = []
    with tarfile.open(tar_filename) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            if m.isfile() and m.name.endswith('.jpg'):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                image_name = os.path.splitext(os.path.basename(m.name))[0]
                image_path = m.name.split('/')
                if image_path[0] == 'data':
                    if image_path[1] == 'train':
                        train_data.append(img)
                    elif image_path[1] == 'val':
                        val_data.append(img)
                    elif image_path[1] == 'test':
                        test_data[image_path[2]].append(img)
                elif image_path[0] == 'targets':
                    if image_path[1] == 'train':
                        for _ in range(3): # Add 3 times the img since there are three versions in the training set
                            train_targets.append(img)
                    elif image_path[1] == 'val':
                        for _ in range(3):
                            val_targets.append(img)
                    elif image_path[1] == 'test':
                        test_targets.append(img)
    # Create tf dataset train_ds
    train_ds = tf.data.Dataset.from_tensor_slices((np.stack(train_data), np.stack(train_targets)))
    train_ds = train_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    #train_ds = train_ds.shuffle(len(train_data))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    # Create tf dataset eval_ds
    eval_ds = tf.data.Dataset.from_tensor_slices((np.stack(val_data), np.stack(val_targets)))
    eval_ds = eval_ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_ds = eval_ds.batch(batch_size)
    eval_ds = eval_ds.cache()
    eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
    # Create test_ds dictionnary that contains all the test sets
    test_targets = np.stack(test_targets)
    test_ds = {}
    for name, data in test_data.items():
        test_ds[name] = tf.data.Dataset.from_tensor_slices((np.stack(data), test_targets))
        test_ds[name] = test_ds[name].map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds[name] = test_ds[name].batch(batch_size)
        test_ds[name] = test_ds[name].cache()
        test_ds[name] = test_ds[name].prefetch(tf.data.experimental.AUTOTUNE)
    return (train_ds, eval_ds, test_ds)