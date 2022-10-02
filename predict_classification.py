from functools import partial
import argparse
import os
import random
import time
import distutils.util
# from PaddleNLP.applications.doc_vqa.Rerank.src.cross_encoder import predict

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import  AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.metrics import MultiLabelsMetric, AccuracyAndF1
import json

from utils import convert_example, create_dataloader, LinearDecayWithWarmup

METRIC_CLASSES = {
    'KUAKE-QIC': Accuracy,
    'KUAKE-QQR': Accuracy,
    'KUAKE-QTR': Accuracy,
    'CHIP-CTC': MultiLabelsMetric,
    'CHIP-STS': MultiLabelsMetric,
    'CHIP-CDN-2C': AccuracyAndF1
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['KUAKE-QIC', 'KUAKE-QQR', 'KUAKE-QTR', 'CHIP-STS', 'CHIP-CTC', 'CHIP-CDN-2C'],
                                 default='KUAKE-QQR', type=str, help='Dataset for sequence classfication tasks.')
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default='gpu', help='Select which device to train model, default to gpu.')
parser.add_argument('--max_seq_length', default=128, type=int, help='The maximum total input sequence length after tokenization.')
parser.add_argument('--init_from_ckpt', default=None, type=str, help='The path of checkpoint to be loaded.')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU/CPU for Inference.')
# parser.add_argument('--enable_SWA',default=False, type=bool, help='Whether enable Stochastic Weight Averaging（SWA）')
parser.add_argument('--SWA_ckpts',default=None,type=str,nargs='+',help='The paths of checkpoints to be used for SWA, tokenizer config will be shared')
parser.add_argument('--output_dir',default='.',type=str)
args = parser.parse_args()


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and compute the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        dataloader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, position_ids, labels = batch
        logits = model(input_ids, token_type_ids, position_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    if isinstance(metric, Accuracy):
        metric_name = 'accuracy'
        result = metric.accumulate()
    elif isinstance(metric, MultiLabelsMetric):
        metric_name = 'macro f1'
        _, _, result = metric.accumulate('macro')
    else:
        metric_name = 'micro f1'
        _, _, _, result, _ = metric.accumulate()

    print('eval loss: %.5f, %s: %.5f' % (np.mean(losses), metric_name, result))
    model.train()
    metric.reset()


    
        
def do_predict():
    paddle.set_device(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.init_from_ckpt)
    
        
    dev_ds,test_ds= load_dataset('cblue',
                        args.dataset,
                        splits=['dev','test'])
    
    # test_ds=load_dataset('cblue',       # comment this line while inferencing on test dataset
    #                     args.dataset,
    #                     splits=['dev'])# for badcase inspecting only
   
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_from_ckpt,
        num_classes=len(test_ds.label_list),
        activation='tanh')

    if  args.SWA_ckpts:    
        base_state_dict= paddle.load(os.path.join(args.init_from_ckpt,'model_state.pdparams'))
        state_keys = {
        x: x.replace('embeddings.', '')
        for x in base_state_dict.keys() if 'embeddings..' in x }
        state_dict_list= [paddle.load(os.path.join(os.path.split(args.init_from_ckpt)[0],
                                                   f"model_{ckpt}",'model_state.pdparams'))
                          for ckpt in args.SWA_ckpts]
        # exclude embedding layers
        for state_dict in state_dict_list:           
            if len(state_keys) > 0:
                state_dict = {
                    state_keys[k]: state_dict[k]
                    for k in state_keys.keys()
                }
        #average model weights
        for key in state_dict.keys(): 
            base_state_dict[key]+=sum([state_dict[key] for state_dict in state_dict_list])
            base_state_dict[key]/=len(state_dict_list)+1
            
        model.set_dict(base_state_dict)
    
    # if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
    #     state_dict = paddle.load(args.init_from_ckpt)          
    #     model.set_dict(state_dict)
        
    model.eval()

    
    trans_func_test = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         is_test=True)
    
    trans_func_train = partial(convert_example,
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        is_test=False)

    batchify_fn_test = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # segment
        Pad(axis=0, pad_val=args.max_seq_length - 1, dtype='int64'),  # position
        # Stack(dtype='int64') #label ;no lablel for test dataset
        ): [data for data in fn(samples)]
    
    batchify_fn_train = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # segment
        Pad(axis=0, pad_val=args.max_seq_length - 1, dtype='int64'),  # position
        Stack(dtype='int64') #label ;no lablel for test dataset
        ): [data for data in fn(samples)]
    
    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn_train,
                                        trans_fn=trans_func_train)
    
    test_data_loader = create_dataloader(test_ds,
                                        mode='test',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn_test,
                                        trans_fn=trans_func_test)
    
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = Accuracy()
    # metric_name = 'accuracy'
    
    evaluate(model, criterion, metric, dev_data_loader)
    
  
    
    
   
    preds_list=[]  
    for step, batch in enumerate(test_data_loader, start=1):
            input_ids, token_type_ids, position_ids= batch
            with paddle.amp.auto_cast(
                    True,
                    custom_white_list=['layer_norm', 'softmax', 'gelu', 'tanh'],
            ):
                    logits = model(input_ids, token_type_ids, position_ids)
                    preds= paddle.argmax(logits,axis=-1)
                    preds_list.append(preds)
    res=np.concatenate(preds_list,axis=0).tolist()
    
    def generate_commit(output_dir, task_name, test_dataset, preds):
        test_examples = test_dataset.data
        pred_test_examples = []
        assert len(test_examples)==len(preds) 
        for idx in range(len(test_examples)):
            example = test_examples[idx]
            example['label']  = preds[idx]
            
            pred_example = {'id': example['id'], 'query1': example['text_a'], 'query2': example['text_b'], 'label': str(example['label'])}
            pred_test_examples.append(pred_example)
        
        with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
            json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)
        
    generate_commit(args.output_dir,'KUAKE-QQR',test_dataset=test_ds,preds=res)
                    
                    
if __name__ == "__main__":
    do_predict()          




