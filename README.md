我们使用了PaddleNLP公开的 [ERNIE-Health](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-health) 在KUAKE_QQR训练集合上微调, 进行模型权重平均后在测试集上最高准确率达到了 86.03

## 1.安装
### 1.1 安装依赖
```bash
conda create --name KUAKE_QQR

conda activate KUAKE_QQR

conda install pip
```
```bash
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

### 1.2 覆盖训练代码
```bash
mv train_classification.py    PaddleNLP/model_zoo/ernie-health/train_classification.py
```

### 1.3 创建预测代码
```bash
mv predict_classification.py  PaddleNLP/model_zoo/ernie-health/predict_classification.py
```
## 2.训练和生成预测结果 
进入脚本所在目录
```bash
cd PaddleNLP/model_zoo/ernie-health/cblue
```
### 2.1 启动训练，自动下载数据集合
```bash
export CUDA_VISIBLE_DEVICES=0 \
&& python train_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--learning_rate 6e-5 \
--epochs 3 \
--seed 1000
```


### 2.2 预测并生成结果

使用单checkpoint模型权重

```bash
export CUDA_VISIBLE_DEVICES=0 \
&& python predict_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--init_from_ckpt "checkpoint/model_900"
```

对多个checkpoint模型权重进行平均后进行预测,`SWA_ckpts`参数为额外使用的进行权重平均的checkpoint训练步数

```bash
export CUDA_VISIBLE_DEVICES=0 \
&& python predict_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--init_from_ckpt "checkpoint/model_900" 
--SWA_ckpts 600 800 1100 1200
```
## Acknowledge
部分代码借鉴了论坛区Alexzhuan给出的[baseline](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.0.0.f7e37785Y4CHqU&postId=409593)，在此对作者和天池社区表示感谢。
