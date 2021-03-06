## Prepare Data

运行 *gen_data.py* 产生随机数据并用pickle导出为文件

## MXNet

> cd mxnet

在mxnet文件夹下运行 *simple_mlp.py* 训练模型并进行结果和网络结构的可视化

## Caffe

> cd caffe

### step 1

在caffe文件夹下运行 *gen_hdf5.py* 将数据转换为HDF5格式

### step 2

使用`caffe`自带的工具来可视化`prototxt`中定义的网络结构：

+ `draw_net.py train.prototxt mlp_train.{png|pdf} --rankdir BT`
+ 使用--rankdir选项从下向上绘制网络

运行 *simple_mlp_train.py* 训练模型，或者

运行 *caffe train -solver solver.prototxt* 训练模型

### step 3

运行 *simple_mlp_test.py* 测试模型及可视化
