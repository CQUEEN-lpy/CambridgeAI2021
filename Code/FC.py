import cv2
import os
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.io as io

# 数据预处理，把图片数据集的所有图片修剪成固定大小形状
def image_tailor(input_dir, out_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # file为root目录中的文件
            filepath = os.path.join(root, file)     # 连接两个或更多的路径名组件，filepath路径为/root/file
            try:
                image = cv2.imread(filepath)        # 根据输入路径读取照片
                dim = (180, 180)                    # 裁剪的尺寸
                resized = cv2.resize(image, dim)    # 按比例将原图缩放成227*227
                path = os.path.join(out_dir, file)  # 保存的路径和相应的文件名
                cv2.imwrite(path, resized)          # 进行保存
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey()
'''
input_patch = 'D:/cambridge2021winter/data/dogs-vs-cats/train/train'  # 数据集的地址
out_patch = 'D:/cambridge2021winter/data/dogs-vs-cats/train180'  # 图片裁剪后保存的地址
image_tailor(input_patch, out_patch)
print('reshape finished')
'''

data_dir = 'E:\\PHD\\HANSHENG\\SRTP\\cam\\data'
data_transform = {x:transforms.Compose([transforms.Scale([64, 64]),
                                       transforms.ToTensor()]) for x in ['train180', 'test180']}
image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                        transform = data_transform[x]) for x in ['train180', 'test180']}
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                           batch_size = 16,
                                           shuffle = True) for x in ['train180', 'test180']}

# 数据预览
'''
X_example, Y_example = next(iter(dataloader['train']))
print(u'X_example个数{}'.format(len(X_example)))
print(u'Y_example个数{}'.format(len(Y_example)))

index_classes = image_datasets['train'].class_to_idx
print(index_classes)

example_classes = image_datasets['train'].classes
print(example_classes)

img = torchvision.utils.make_grid(X_example)
img = img.numpy().transpose([1,2,0])
print([example_classes[i] for i in Y_example])
plt.imshow(img)
plt.show()
'''

# 模型搭建  简化了的VGGnet
class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64*64*3,64*64*64),
            torch.nn.ReLU,
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(64*64*64,32*32*64),
            torch.nn.ReLU,
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(32*32*64,16*16*128),
            torch.nn.ReLU,
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(16*16*128,8*8*256),
            torch.nn.ReLU,
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(8*8*256, 4*4*512),
            torch.nn.ReLU,
            torch.nn.Dropout(p=0.5),
        )
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 64 * 3, 4 * 4 * 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),)

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 2))


    def forward(self, inputs):
        inputs = inputs.view(-1, 64 * 64 * 3)
        x = self.dense(inputs)
        x = self.Classes(x)
        return x

model = Models()
print(model)


####训练模型
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

Use_gpu = torch.cuda.is_available()
print(Use_gpu)

if Use_gpu:
    model = model.cuda()

epoch_n = 50
time_open = time.time()
train_accuracy = torch.zeros([50000,1])
test_accuracy = torch.zeros([50000,1])

iter_train = 0
iter_test = 0
for epoch in range(epoch_n):
    print('epoch{}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)
    torch.save(model.state_dict(), "saved_models_FC/model_%d.pth" % epoch)
    print('saving model finished')
    for phase in ['train180', 'test180']:
        if phase == 'train180':
            print('training...')
            model.train(True)
        else:
            print('validing...')
            model.train(False)
        running_loss = 0.0
        running_corrects = 0.0

        for batch, data in enumerate(dataloader[phase], 1):
            X, Y = data

            X, Y = Variable(X).cuda(), Variable(Y).cuda()

            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()

            loss = loss_f(y_pred, Y)

            if phase == 'train180':
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_corrects += torch.sum(pred == Y.data)

            if batch % 500 == 0 and phase == 'train180':
                print('batch{},trainLoss;{:.4f},trainAcc:{:.4f}'.format(batch, running_loss / batch,
                                                                        100 * running_corrects / (16 * batch)))

            epoch_loss = running_loss * 16 / len(image_datasets[phase])
            epoch_acc = 100 * running_corrects / len(image_datasets[phase])



            print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))

            time_end = time.time() - time_open
            print(time_end)

        if phase == 'train180':
            train_accuracy[iter_train] = epoch_acc.data
            iter_train = iter_train + 1
        else:
            test_accuracy[iter_test] = epoch_acc.data
            iter_test = iter_test + 1

#save accuracy in the form of EXCEL
train_accuracy = train_accuracy.cpu()
train_accuracy = train_accuracy.numpy()
np.savetxt('train_accuracy_FC', train_accuracy)

test_accuracy = test_accuracy.cpu()
test_accuracy = test_accuracy.numpy()
np.savetxt('test_accuracy_FC', test_accuracy)


