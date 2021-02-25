import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        # print(image.size)


if __name__ == '__main__':
    s = SimplePreprocessor(32, 32)
    img = cv2.imread('/home/xxjian/DeepLearningMaterial/kagglecatsanddogs_3367a/PetImages/Cat/9759.jpg')
    # print(img)
    cv2.imshow('src', img)
    cv2.imshow("resize", s.preprocess(img))
    # print(img.size)
    cv2.waitKey(0)
    # cv2.destroyallWindows()