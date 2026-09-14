import random
# 写txt
import os
import tqdm

train_imgpath = '../data/GLAS/train/images'
test_imgpath = '../data/GLAS/test/images'
txtsavepath = '../data/GLAS/'

imgtxt_1 = open('../data/GLAS/train.txt', 'w')
imgtxt_2 = open('../data/GLAS/val.txt', 'w')
imgtxt_3 = open('../data/GLAS/test.txt', 'w')

n = 0
train_image_list = os.listdir(train_imgpath)
random.shuffle(train_image_list)  # 使用shuffle()函数打乱原始列表
for img in tqdm.tqdm(train_image_list):
    if n < (0.8 * len(train_image_list)):
        name = img[0:-4] + '\n'
        imgtxt_1.write(name)
        n = n + 1
    else:
        name = img[0:-4] + '\n'
        imgtxt_2.write(name)
        n = n + 1

test_image_list = os.listdir(test_imgpath)
random.shuffle(test_image_list)  # 使用shuffle()函数打乱原始列表
for img in tqdm.tqdm(test_image_list):
    name = img[0:-4] + '\n'
    imgtxt_3.write(name)
