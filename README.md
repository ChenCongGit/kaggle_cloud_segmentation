# Understanding Clouds from Satellite Images
[Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization)比赛代码，最终成绩铜牌，Public Leaderboard:「129/1556」，Private Leaderboard:「138/1538」，dice: 65.563%

## Overview
该比赛对下图这种气象云图进行分割，将不同形态的云区域分割出来，包括sugar、flower、fish、gravel。
![](/相关图片/气象云图.gif "气象云图")

## 数据构成
我们共有9244张云图图像，其中5546张标注训练集，3698张无标注测试集。我们将训练数据集均匀划分为5折，采用5折交叉验证训练、验证和选择后处理参数。测试集中的25%作为Public Leaderboard，另75%为Private Leaderboard，作为最终比赛结果。

## 算法细节
训练集气象云图的标签非常粗略（大部分简单的是矩形，多边形），标记很不准确，因此仅仅使用单一的分割网络模型很难获得很高的预测结果。
### 模型训练与测试
 - 交叉验证: 我们采用5折交叉验证，训练5个不同的模型，分别预测，最后进行mask投票。
 - 训练多个模型: 我们以'se_resnext101'为骨架，采用imagenet预训练参数，分别训练了FPN和Unet网络。
 - 模型集成: 我们训练了FPN和Unet网络，且每个网络采用交叉验证训练了5个模型，共10个模型，首先对两个模型的同一折logits平均融合集成，最后对不同折预测结果投票集成。
 - 损失函数: 我们使用二分类交叉熵和dice损失函数，进行4个二分类过程（4个类别分割区域可以重叠）。
 - 学习过程: 我们采用ReduceLROnPlateau学习方案，当验证集loss在4个epoch之内不下降，将学习率下降到0.35倍。
 - 数据增强: 水平竖直翻转，随机旋转，随机rescale，平移，GridDistortion，resize到固定大小(384, 576)
 - 测试时增强（tta）: 水平竖直翻转，多尺度（5/6, 1, 7/6）

### 后处理过程
 - 二值化: 选择不同阈值对网络预测结果二值化为mask
 - 去除小区域。
 - 凸包处理: opencv凸包处理，简化区域形状
 - 后处理参数网格搜索。

## 代码
### 5折训练
CUDA_VISIBLE_DEVICES = 0 python seg.py
### 5折交叉验证
CUDA_VISIBLE_DEVICES = 0 python seg.py
### 模型融合
CUDA_VISIBLE_DEVICES = 0 python seg_average.py
### 预测结果集成
python ensemble.py


