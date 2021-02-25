import numpy as np
import cv2
import os


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        # print(imagePaths)
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath[0:2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    if (image is None):
                        print(i)
                        os.remove(imagePaths[i])
                        print('file: ')
                        print(imagePaths[i])
                        print('is removed.')
                        continue
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))


if __name__ == '__main__':
    imagePaths = 'D:\pycharm\knnClassifier\test_to_ZJY\test_to_ZJY'
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    # (data, labels) = sdl.load(imagePaths, verbose=10)
    # data = data.reshape((data.shape[0], 3072))