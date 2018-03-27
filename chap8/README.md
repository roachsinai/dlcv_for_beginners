## Prepare Data
### step 1

> cd data
>
> bash ./download_mnist.sh

下载mnist.pkl.gz，然后运行 *convert_mnist.py* 将pickle格式的数据转换为图片

如果原链接无法下载可以到下面云盘地址下载：
http://pan.baidu.com/s/1bHmm7s

> py convert_mnist.py

### step 2
> python gen_caffe_imglist.py mnist/train train.txt

> python gen_caffe_imglist.py mnist/val val.txt

> python gen_caffe_imglist.py mnist/test test.txt

得到图片的列表，然后运行： 
> convert_imageset ./ train.txt train_lmdb --gray --shuffle

> convert_imageset ./ val.txt val_lmdb --gray --shuffle

> convert_imageset ./ test.txt test_lmdb --gray --shuffle

产生lmdb

### step 3
> python gen_mxnet_imglist.py mnist/train train.lst

> python gen_mxnet_imglist.py mnist/val val.lst

> python gen_mxnet_imglist.py mnist/test test.lst

用于产生图片的列表，然后运行（默认生成对应于*.lst的，*.idx和*.rec）

> im2rec.py train.lst ./ --color 0

> im2rec.py test.lst ./ --color 0

> im2rec.py test.lst ./ --color 0

产生ImageRecordio文件

---

## MXNet

`mxnet`根据代码中的`ctx`设置选择是否在`GPU`上运行。

> cd mxnet

+ 运行 *train_lenet5.py* 训练模型
  + `py train_lenet5.py`
  + 代码中使用的数据增加，所以相对`caffe`准确率有提升，执行时间同样增涨
+ 运行 *score_model.py* 在测试集上评估模型
  + `py score_model.py`
  + 得益于数据增加，`acc`较`caffe`有提升
+ 运行 *benchmark_model.py* 测试模型性能
  + `py benchmark_model.py`
  + 使用未加数据扰动时，比`caffe`快3ms（mxnet在我的asus笔记本是5ms）
+ 运行 *recognize_digit.py* 跟图片路径作为参数用于手写数字图片识别
  + `py recognize_digit.py ../data/mnist/test`

---

## Caffe

+ *lenet_train_val.prototxt* & *lenet_train_val_aug.prototxt*
  + 分别是用原始数据和增加后数据训练模型的网络结构和数据定义文件
+ *lenet_solver.prototxt* & *lenet_solver_aug.prototxt* 
  + 分别是训练原始数据和增加后数据的solver文件
+ *lenet_test.prototxt* 是用于在测试数据上测试模型的网络结构和数据源定义文件
+ *lenet.prototxt* 是用于部署的网络结构定义文件
+ 运行 *recognize_digit.py* 接测试文件的列表用来演示手写数字图片识别

> cd caffe

训练LeNet5

> caffe train -solver lenet_solver.prototxt -gpu 0

测试模型acc

> caffe test -model lenet_test.prototxt -weights mnist_lenet_iter_36000.caffemodel -gpu 0 -iterations 100

模型性能评估

GPU

> caffe time -model lenet.prototxt -gpu 0

CPU

> caffe time -model lenet.prototxt

手写数字识别

> py recognize_digit.py ../data/train.txt
