"""Data generator for center cropped and transformed MNIST images

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np
from skimage.transform import resize, rotate


class DataGenerator(Sequence):
    def __init__(self,
                 args,
                 shuffle=True,
                 siamese=False,
                 mine=False,
                 crop_size=4):
        """Multi-threaded data generator. Each thread reads
            a batch of images and performs image transformation
            such that the image class is unaffected

        Arguments:
            args (argparse): User-defined options such as
                batch_size, etc
            shuffle (Bool): Whether to shuffle the dataset
                before sampling or not
            siamese (Bool): Whether to generate a pair of 
                image (X and Xbar) or not
            mine (Bool): Use MINE algorithm instead of IIC
            crop_size (int): The number of pixels to crop
                from all sides of the image
        """
        self.args = args
        self.shuffle = shuffle
        self.siamese = siamese
        self.mine = mine
        self.crop_size = crop_size
        self._dataset()
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch
        """
        return int(np.floor(len(self.indexes) / self.args.batch_size))


    def __getitem__(self, index):
        """Image sample Indexes for the current batch
        """
        start_index = index * self.args.batch_size
        end_index = (index+1) * self.args.batch_size
        return self.__data_generation(start_index, end_index)

    def _dataset(self):
        """Load dataset and normalize it
        """
        dataset = self.args.dataset
        if self.args.train:
            (self.data, self.label), (_, _) = dataset.load_data()
        else:
            (_, _), (self.data, self.label) = dataset.load_data()

        if self.args.dataset == mnist:
            self.n_channels = 1
        else:
            self.n_channels = self.data.shape[3]

        image_size = self.data.shape[1]
        side = image_size - self.crop_size
        self.input_shape = [side, side, self.n_channels]

        # from sparse label to categorical
        self.n_labels = len(np.unique(self.label))
        self.label = to_categorical(self.label)

        # reshape and normalize input images
        orig_shape = [-1, image_size, image_size, self.n_channels]
        self.data = np.reshape(self.data, orig_shape)
        self.data = self.data.astype('float32') / 255
        self.indexes = [i for i in range(self.data.shape[0])]


    def on_epoch_end(self):
        """If opted, shuffle dataset after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def random_crop(self, image, target_shape, crop_sizes):
        """Perform random crop, resize back to its target shape

        Arguments:
            image (tensor): Image to crop and resize
            target_shape (tensor): Output shape
            crop_sizes (list): A list of sizes the image 
                can be cropped
        """
        height, width = image.shape[0], image.shape[1]
        crop_size_idx = np.random.randint(0, len(crop_sizes))
        d = crop_sizes[crop_size_idx]
        x = height - d
        y = width - d
        center = np.random.randint(0, 2)
        if center:
            dx = dy = d // 2
        else:
            dx = np.random.randint(0, d + 1)
            dy = np.random.randint(0, d + 1)

        image = image[dy:(y + dy), dx:(x + dx), :]
        image = resize(image, target_shape)
        return image


    def random_rotate(self,
                      image, 
                      deg=20, 
                      target_shape=(24, 24, 1)):
        """Random image rotation

        Arguments:
            image (tensor): Image to crop and resize
            deg (int): Degrees of rotation
            target_shape (tensor): Output shape
        """
        angle = np.random.randint(-deg, deg)
        image = rotate(image, angle)
        image = resize(image, target_shape)
        return image


    def __data_generation(self, start_index, end_index):
        """Data generation algorithm. The method generates
            a batch of pair of images (original image X and
            transformed imaged Xbar). The batch of Siamese
            images is used to trained MI-based algorithms:
            1) IIC and 2) MINE (Section 7)

        Arguments:
            start_index (int): Given an array of images,
                this is the start index to retrieve a batch
            end_index (int): Given an array of images,
                this is the end index to retrieve a batch
        """

        d = self.crop_size // 2
        crop_sizes = [self.crop_size*2 + i for i in range(0,5,2)]
        image_size = self.data.shape[1] - self.crop_size
        x = self.data[self.indexes[start_index : end_index]]
        y1 = self.label[self.indexes[start_index : end_index]]

        target_shape = (x.shape[0], *self.input_shape)
        x1 = np.zeros(target_shape)
        if self.siamese:
            y2 = y1 
            x2 = np.zeros(target_shape)

        for i in range(x1.shape[0]):
            image = x[i]
            x1[i] = image[d: image_size + d, d: image_size + d]
            if self.siamese:
                rotate = np.random.randint(0, 2)
                # 50-50% chance of crop or rotate
                if rotate == 1:
                    shape = target_shape[1:]
                    x2[i] = self.random_rotate(image,
                                               target_shape=shape)
                else:
                    x2[i] = self.random_crop(image,
                                             target_shape[1:],
                                             crop_sizes)

        # for IIC, we are mostly interested in paired images
        # X and Xbar = G(X)
        if self.siamese:
            # If MINE Algorithm is chosen, use this to generate
            # the training data (see Section 9)
            if self.mine:
                y = np.concatenate([y1, y2], axis=0)
                m1 = np.copy(x1)
                m2 = np.copy(x2)
                np.random.shuffle(m2)

                x1 =  np.concatenate((x1, m1), axis=0)
                x2 =  np.concatenate((x2, m2), axis=0)
                x = (x1, x2)
                return x, y

            x_train = np.concatenate([x1, x2], axis=0)
            y_train = np.concatenate([y1, y2], axis=0)
            y = []
            for i in range(self.args.heads):
                y.append(y_train)
            return x_train, y

        return x1, y1


if __name__ == '__main__':
    datagen = DataGenerator()

