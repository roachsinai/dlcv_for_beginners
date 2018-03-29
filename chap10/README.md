## 第十章 迁移学习和模型微调

从搜索引擎上下载关键词对应的图片，并利用模型微调进行训练，以及最后的模型评估，分析和可视化。

具体用法参考本书第十章内容。

书中例子对应的预训练模型可以在下面地址下载：  
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/food_resnet-10_iter_10000.caffemodel  
或  
http://pan.baidu.com/s/1jHRLsLw

## 准备数据

```python

cd data

# 收集图片
py collect_data.py

# 去除无效（opencv无法打开）图片
py remove_invalid_images.py ./

# 去除重复图片
# 去除完全相同的图片
pacman -S fdupes
fdupes -rdN ./
# 去除视觉上相似的图片
yaourt -S findimagedupes

mkdir train
mv 00? train
py downscale.py train 256
findimagedupes -R train > dup_list
py remove_dups_from_list.py dup_list

# 抽出测试集
py sample_val.py

# 生成训练数据
# 数据增加， 或者执行 bash link_data_augmentation.sh
ln -s ../../chap6/data_augmentation/run_augmentation.py run_augmentation.py
ln -s ../../chap6/data_augmentation/image_augmentation.py image_augmentation.py
# 执行数据增加
py food_augmentation.py

# 生成lmdb
py gen_label_list.py train
py gen_label_list.py val

convert_imageset ./ train.txt train_lmdb -resize_width 224 -resize_height 224 --shuffle
convert_imageset ./ val.txt val_lmdb -resize_width 224 -resize_height 224 --shuffle

```

## Finetune

> [下载已经训练好的ResNet-10：resnet10_cvgj_iter_320000.caffemodel](https://github.com/cvjena/cnn-models/blob/master/ResNet_preact/ResNet10_cvgj/model_download_link.txt)

```python
# 使用caffe进行模型微调
# use -log_dir ./ get the error: ERROR: unknown command line flag 'log_dir'
# -a mean append
caffe train -solver solver.prototxt -weights resnet10_cvgj_iter_320000.caffemodel | tee -a log.txt

# 根据caffe做出的准确率变化的图示，选择test loss最低的时候， iter = 10000
# 使用训练10000次的模型进行预测
py recognize_food.py val.txt
```

## Confuse Matrix

画七类的混淆矩阵，
`py make_confusion_matrix.py`

在混淆矩阵中，如果类别之间样本数量差距较大，可以进行归一化（在代码中设置是否进行归一化）

## 画P-R曲线和ROC曲线

```python
# PR
py sort_kaoya_by_pred_prob.py
# ROC 
py kaoya_shuizhurou_roc_auc.py
```

对于二分类模型，

+ F-measure对应与P-R曲线，可以挑选使得F值最大的阈值作为分类边界
+ 使用ROC曲线，则是和（0, 1）（1, 0）这两个点的连线的交点对应的预测值作为分类边界