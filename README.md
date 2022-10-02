使用PaddleNLP开源医学预训练模型'ernie-health-chinese'在KUAKE_QQR训练数据集合上微调,进行模型权重平均后在测试集上准确率达到86.03

## 安装环境依赖
conda create --name KUAKE_QQR
conda activate KUAKE_QQR
conda install pip
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

git clone https://github.com/PaddlePaddle/PaddleNLP.git

## 覆盖训练代码

mv train_classification.py    PaddleNLP/model_zoo/ernie-health/train_classification.py

## 创建预测代码

mv predict_classification.py  PaddleNLP/model_zoo/ernie-health/predict_classification.py


## 进入模型所在文件夹运行训练脚本

cd PaddleNLP/model_zoo/ernie-health/cblue

export CUDA_VISIBLE_DEVICES=0 \
&& python train_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--learning_rate 6e-5 \
--epochs 3 \
--seed 1000



## 使用单checkpoint模型权重预测
export CUDA_VISIBLE_DEVICES=0 \
&& python predict_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--init_from_ckpt "/home/credog/KUAKE_QQR/PaddleNLP/model_zoo/ernie-health/cblue/checkpoint/model_900"


## 对多个checkpoint模型权重进行平均后进行预测,SWA_ckpts参数为使用的checkpoint训练步数
export PNLP_ROOT="/home/credog/KUAKE_QQR/PaddleNLP" \
&& export CUDA_VISIBLE_DEVICES=0 \
&& python predict_classification.py \
--dataset KUAKE-QQR \
--batch_size 32 \
--max_seq_length 64 \
--init_from_ckpt "/home/credog/KUAKE_QQR/PaddleNLP/model_zoo/ernie-health/cblue/checkpoint/model_900" 
--SWA_ckpts 600 800 1100 1200
