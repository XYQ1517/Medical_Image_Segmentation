import random
# 写txt
import os
import tqdm


imgpath = '../data/PH2/images'
txtsavepath = '../data/PH2/'

imgtxt_1 = open('../data/PH2/train.txt', 'w')
imgtxt_2 = open('../data/PH2/val.txt', 'w')
imgtxt_3 = open('../data/PH2/test.txt', 'w')

n = 0
image_list = os.listdir(imgpath)
random.shuffle(image_list)  # 使用shuffle()函数打乱原始列表
for img in tqdm.tqdm(image_list):
    name = img[0:-4] + '\n'
    imgtxt_3.write(name)
    n = n + 1

imgtxt_1.close()
imgtxt_2.close()
imgtxt_3.close()

# imgpath = '../data/PH2/images'
# txtsavepath = '../data/PH2/'
#
# imgtxt_1 = open('../data/PH2/train.txt', 'w')
# imgtxt_2 = open('../data/PH2/val.txt', 'w')
# imgtxt_3 = open('../data/PH2/test.txt', 'w')
#
# n = 0
# image_list = os.listdir(imgpath)
# random.shuffle(image_list)  # 使用shuffle()函数打乱原始列表
# for img in tqdm.tqdm(image_list):
#     if n < (0.7 * len(image_list)):
#         name = img[0:-4] + '\n'
#         imgtxt_1.write(name)
#         n = n + 1
#     elif 0.7 * len(image_list) <= n < 0.8 * len(image_list):
#         name = img[0:-4] + '\n'
#         imgtxt_2.write(name)
#         n = n + 1
#     else:
#         name = img[0:-4] + '\n'
#         imgtxt_3.write(name)
#         n = n + 1
#
# imgtxt_1.close()
# imgtxt_2.close()
# imgtxt_3.close()
#
