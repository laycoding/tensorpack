#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import os
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger

from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.utils.fs import mkdir_p

SIZE = 224

class HoMGRMeta(object):
	"""
	Provide methods to access metadata for HoMGR dataset.
	HoMGR dataset
	├── train
		├── image_tr.txt
		├── train00001.jpg
		├── train00002.jpg
		├── train00003.jpg
		...
		...
		├── train16763.jpg

	"""

	def __init__(self, dir=None):
		if dir is None:
			print "plz input the dataset path"
		self.dir = dir
		mkdir_p(self.dir)
		f = os.path.join(self.dir, 'image_tr.txt')
		if not os.path.isfile(f):
			print "can not find image_tr.txt"
		self.caffepb = None

	def get_synset_words_3(self):
		pass

	def get_synset_3(self):
		pass

	def get_image_list(self, name, dir_structure='train'):
		"""
		Args:
			name (str): 'train' or 'val' or 'test'
			dir_structure (str): same as in :meth:'HoMGR.__init__()'.
		Returns:
			list: list of (image filename, label)
		"""
		assert name in ['train', 'val', 'test']
		# assert dir_structure in ['original', 'train']

		with open(self.dir + os.sep + name + '/image_' + name + '.txt') as dict_map:
			ret = []
			lines = dict_map.readlines()
			for name_label in lines[1:]:
				name = name_label.strip().split(',')[0]
				label = int(name_label.strip().split(',')[1]) - 1

				ret.append((name, label))
		assert len(ret)
		return ret

class HoMGRFiles(RNGDataFlow):
	def __init__(self, dir, name, meta_dir=None, shuffle=None, dir_structure=None):
		assert name in ['train', 'test', 'val'], name
		assert os.path.isdir(dir), dir
		self.full_dir = os.path.join(dir, name)

		if shuffle is None:
			shuffle = name == 'train'
		self.shuffle = shuffle

		if name == 'train':
			dir_structure = 'train'
		
		meta = HoMGRMeta(dir)
		self.imglist = meta.get_image_list(name, dir_structure)

		for fname, _ in self.imglist[:10]:
			fname = os.path.join(self.full_dir, fname)
			assert os.path.isfile(fname), fname

	def size(self):
		return len(self.imglist)

	def get_data(self):
		idxs = np.arange(len(self.imglist))
		if self.shuffle:
			self.rng.shuffle(idxs)
		for k in idxs:
			fname, label = self.imglist[k]
			fname = os.path.join(self.full_dir, fname)
			yield [fname, label]

class HoMGR(HoMGRFiles):
	"""
	Produce uint8 HoMGR images of shape [h, w, 3(BGR)], and a label between [0,2]
	"""
	def __init__(self, dir, name, meta_dir=None, shuffle=None, dir_structure=None):
		super(HoMGR, self).__init__(
			dir, name, meta_dir, shuffle, dir_structure)
		self.im_size = (SIZE, SIZE)

	def get_data(self):
		for fname, label in super(HoMGR, self).get_data():
			im = cv2.imread(fname, cv2.IMREAD_COLOR)
			im = cv2.resize(im, self.im_size)
			assert im is not None, fname
			yield [im, label]

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out

def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(      # Remove these augs if your CPU is not fast enough
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors

def get_HoMGR_dataflow(
        datadir, name, batch_size,
        augmentors, parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if isTrain:
        ds = HoMGR(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = HoMGRFiles(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_HoMGR(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))
    

class ImageNetModel(ModelDesc):
    weight_decay = 1e-4
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Whether to apply weight decay on BN parameters.
    """
    weight_decay_on_bn = False

    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = ImageNetModel.image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(logits, label)

        if self.weight_decay > 0:
            if self.weight_decay_on_bn:
                pattern = '.*/W|.*/gamma|.*/beta'
            else:
                pattern = '.*/W'
            wd_loss = regularize_cost(pattern, tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)
        return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``
        Returns:
            Nx1000 logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    @staticmethod
    def image_preprocess(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss
