import sys
import numpy as np
import chainer
import random
import scipy.misc

class_to_index = dict()
for i, c in enumerate(range(1, 201)):
    class_to_index[c] = i


class Cub2011EncDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path_input, path_truth, root_input, root_truth, size=128, resize_method='bilinear', augmentation=False, crop_ratio=0.9):
        self.base_input = chainer.datasets.LabeledImageDataset(path_input, root_input)
        self.base_truth = chainer.datasets.LabeledImageDataset(path_truth, root_truth)
        self.size = size
        self.resize_method = resize_method
        self.augmentation = augmentation
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.base_input)

    def transform(self, image1, image2):
        _, h, w = image1.shape
        if image1.shape[0] == 1:
            image1 = np.concatenate([image1, image1, image1], axis=0)
        if image2.shape[0] == 1:
            image2 = np.concatenate([image2, image2, image2], axis=0)
        if image1.shape != image2.shape:
            raise Exception("input and truth image mismatch")

        short_side = h if h < w else w
        if self.augmentation:
            crop_size = int(short_side * self.crop_ratio)
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image1 = image1[:, :, ::-1]
                image2 = image2[:, :, ::-1]
        else:
            crop_size = short_side
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image1 = image1[:, top:bottom, left:right]
        image2 = image2[:, top:bottom, left:right]
        _, h, w = image1.shape
        image1 = scipy.misc.imresize(image1.transpose(1, 2, 0),
                                    [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        image2 = scipy.misc.imresize(image2.transpose(1, 2, 0),
                                    [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        image1 = image1 / 128. - 1.
        image1 += np.random.uniform(size=image1.shape, low=0., high=1. / 128)
        image2 = image2 / 128. - 1.
        image2 += np.random.uniform(size=image1.shape, low=0., high=1. / 128)
        return image1, image2


    def get_example(self, i):
        image_input, label = self.base_input[i]
        image_truth, _ = self.base_truth[i]
        image_input, image_truth = self.transform(image_input, image_truth)
        index = class_to_index[int(label)]
        return image_input, image_truth, index


# python cub.py ROOT_PATH
if __name__ == "__main__":
    import glob, os, sys

    root_path = sys.argv[1]

    dirname_to_label = {}
    with open('dirname_to_label_cub.txt', 'r') as f:
        for line in f:
            dirname, label = line.strip('\n').split()
            dirname_to_label[dirname] = label

    count = 0
    n_image_list = []
    filenames = glob.glob(root_path + '/*/*.jpg')
    for filename in filenames:
        filename = filename.split('/')
        dirname = filename[-2]
        label = int(dirname_to_label[dirname])
        n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
        count += 1
        if count % 10000 == 0:
            print(count)
    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt('image_list_cub.txt', n_image_list, fmt="%s")
