# Importing the required dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag

import os
import numpy as np
import argparse
import json

from tqdm import tqdm

import sys
sys.path.insert(0, 'Utils/')
from JSON_read import read_data

def eval(attrb,idx2tag,y_true,y_pred,num):
        # Metricsidx2tag
        gold = len([idx2tag[hat] for hat in y_true[y_true==num]])
        prop = len([idx2tag[hat] for hat in y_pred[y_pred==num]])
        correct = (np.logical_and(y_true==y_pred, y_true==num)).astype(np.int).sum()

        try:
            precision = float(correct / prop)
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = float(correct / gold)
        except ZeroDivisionError:
            recall = 1.0

        try:
            f1 = float(2*precision*recall / (precision + recall))
        except ZeroDivisionError:
            if precision*recall==0:
                f1=1.0
            else:
                f1=0
        print(f"\n{attrb} |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

def test_model(model, iterator, fpath):
    '''
    Tests the model on the test data after loading
    the checkpoints from the local directory.

    '''

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x,
            y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # Writing the predicted data into a json file.
    test_data = []
    print("\n--------------Evaluating the metrics----------------")
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            tmp = {'Text': words[5:-5],
                   'Tags': "  "+tags[5:-5],
                   'predict': " ".join(preds[1:-1])}
            test_data.append(tmp)
            fout.write("\n")
    json_data = {'Test data':test_data}
    with open('shopee_data/JSON_files/test_data.json','w') as f:
        json.dump(json_data, f, indent=2)


    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    # Metrics for Fit Type:FT
    ft_gold = len([idx2tag[hat] for hat in y_true[y_true==2]])
    ft_prop = len([idx2tag[hat] for hat in y_pred[y_pred==2]])
    ft_correct = (np.logical_and(y_true==y_pred, y_true==2)).astype(np.int).sum()

    try:
        precision = float(ft_correct / ft_prop)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = float(ft_correct / ft_gold)
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = float(2*precision*recall / (precision + recall))
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    print("\nFT |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

    # Metrics for Collar :C
    c_gold = len([idx2tag[hat] for hat in y_true[y_true==3]])
    c_prop = len([idx2tag[hat] for hat in y_pred[y_pred==3]])
    c_correct = (np.logical_and(y_true==y_pred, y_true==3)).astype(np.int).sum()
    try:
        precision = c_correct / c_prop
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = c_correct / c_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    print("\nC  |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

    # Metrics for Pattern : P
    p_gold = len([idx2tag[hat] for hat in y_true[y_true==4]])
    p_prop = len([idx2tag[hat] for hat in y_pred[y_pred==4]])
    p_correct = (np.logical_and(y_true==y_pred, y_true==4)).astype(np.int).sum()
    try:
        precision = p_correct / p_prop
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = p_correct / p_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    print("\nP  |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

    # Metrics for Material : M
    m_gold = len([idx2tag[hat] for hat in y_true[y_true==5]])
    m_prop = len([idx2tag[hat] for hat in y_pred[y_pred==5]])
    m_correct = (np.logical_and(y_true==y_pred, y_true==5)).astype(np.int).sum()
    try:
        precision = m_correct / m_prop
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = m_correct / m_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    print("\nM  |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

    # Metrics for Sleeves: S
    s_gold = len([idx2tag[hat] for hat in y_true[y_true==6]])
    s_prop = len([idx2tag[hat] for hat in y_pred[y_pred==6]])
    s_correct = (np.logical_and(y_true==y_pred, y_true==6)).astype(np.int).sum()
    try:
        precision = s_correct / s_prop
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = s_correct / s_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    print("\nS  |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")

    #Metrics for the overall test data
    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    #print(f"num_proposed:{num_proposed}")
    #print(f"num_correct:{num_correct}")
    #print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    USER_INPUT = str(input('Instance creation [NAME TO THIS TEST]: '))
    final = fpath + USER_INPUT + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    print("\n\nTest data |  Precision=%.4f"%precision, "|  Recall=%.4f"%recall, "|  F1=%.4f"%f1,"|")
    print("\nLog data being written...")
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")
    read_data('shopee_data/JSON_files/test_data.json')
    print("Completed.")
    os.remove("temp")
    print("----------------------------------------------------")


def load_checkpoint(model, PATH):

    print('\n[~]Testline...')
    print('[~]Loading the checkpoints...')
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint)
    print('[~]Model loaded.')
    print('[~]State_dict loaded\n')
    print(model)
    model.eval()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--logdir", type=str, default="shopee_data/LOGS/")
    parser.add_argument("--checkpoint_path", type=str, default="/media/prodigy/DATA/model/Before_labelling/3.pt")
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--test_data", type=str, default="/home/prodigy/Desktop/bert_ner/shopee_data/TEST_DATA/test_chang_version.txt")
    hp = parser.parse_args()

    DEVICE = 'cpu'

    model = Net(vocab_size = len(VOCAB), device=DEVICE, finetuning=hp.finetuning)
    model = load_checkpoint(model, hp.checkpoint_path)

    test_dataset = NerDataset(hp.test_data)
    test_iter = data.DataLoader(dataset = test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    fpath = hp.logdir
    print("--------------Starting to test data-----------------\n")
    test_model(model,test_iter,fpath)
