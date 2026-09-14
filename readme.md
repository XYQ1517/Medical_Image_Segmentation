## 1.配置python环境
    
PyTorch使用2.1版本即可，其余的缺啥装啥(缺少的库可以等后续执行代码报错的时候再安装)。

## 2.下载数据集

下载Kvasir-seg等数据集，并按照如下示例存放：

    data
    └── Kvasir
        ├── images
        └── masks
    └── BUSI
        ├── images
        └── masks
    └── GLAS
        ├── images
        └── masks
    └── ISIC2018
        ├── images
        └── masks

## 3. 数据集预处理

利用"./process/trans_Kvasir.py"等文件对Kvasir数据集进行预处理，得到train.txt、val.txt和test.txt：

    data
    └── Kvasir
        ├── images
        ├── masks
        ├── train.txt
        ├── val.txt
        └── test.txt

## 4. Train
在train.py中修改模型及相关参数后直接执行train.py文件即可。

## 5. Test
test.py中修改模型及相关参数后直接执行test.py文件即可。

## 6. Note
1. modeling中存放了许多对比实验的方法，按照步骤四和五去修改对应的模型参数即可去复现其他模型在本框架下的对比实验。
2. train.py和test.py中同时加载了很多方法，不需要的话可以删去。
3. modeling里面部分模型会用到预训练权重需要自行下载。