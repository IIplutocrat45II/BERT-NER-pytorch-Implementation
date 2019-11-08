import pandas as pd
import os
from tqdm import tqdm

THRESHOLD_PATH = r"tmp_files/above_thresholds.txt"
JSON_PATH = r"shopee.json"

threshold_txt = open(THRESHOLD_PATH,'r').read()
json_data     = pd.read_json(JSON_PATH)[33]

MAP_DATA = {}
KEY = {'Material': 'M',
       'Collar'  : 'C',
       'Neckline': 'C',
       'Sleeves' : 'S',
       'Pattern' : 'P',
       'Fit Type': 'FT'}


def read_train_data(TRAIN_TXT_PATH,SAMPLE_PATH):

    train_txt = open(TRAIN_TXT_PATH, 'r+').read().strip().split("\n\n")
    sample_txt = open(SAMPLE_PATH,'w')

    for entry in tqdm(train_txt):
        words = [line.split()[0] for line in entry.splitlines()]
        tags = ([line.split()[1] for line in entry.splitlines()])
        for key,val in MAP_DATA.items():
            if key in words:
                for i,word in enumerate(words):
                    if word == key:
                        tags[i] = MAP_DATA[key]
        for w,t in zip(words,tags):
            str_ = w + ' O ' + 'O ' + t + '\n'
            sample_txt.writelines(str_)
        sample_txt.writelines('\n')
    print("[~]Writing the data into the sample.txt")


def mapper(pos_, map_, pair):
    print("[~]Mapping words to the attributes...")
    for key, val in json_data.items():
        for key_, val_ in val.items():
            for res in val_:
                if res in map_:
                    for tmp in map_:
                        if res == tmp:
                            MAP_DATA.update({pair[tmp]:KEY[key.strip()]})
    return True


if __name__ == '__main__':

    new_words = threshold_txt.split('\n\n')[:-1]

    pos_ = [line.split('\n')[0] for line in new_words]
    map_ = [line.split('\n')[1] for line in new_words]
    pair = {b:a for a,b in zip(pos_,map_)}

    if mapper(pos_,map_, pair) == True:
        #Train
        TRAIN_TXT_PATH = r"/home/prodigy/Desktop/bert_ner/shopee_data/train.txt"
        SAMPLE_PATH  = r"sample.txt"

        read_train_data(TRAIN_TXT_PATH,SAMPLE_PATH)
        os.remove("/home/prodigy/Desktop/bert_ner/shopee_data/train.txt")
        print('[~]train.txt removed.')
        os.rename('sample.txt','/home/prodigy/Desktop/bert_ner/shopee_data/train.txt')
        print('[~]Sample.txt renamed to train.txt to moved to the shopee_data directory.')


        #Valid
        #TRAIN_TXT_PATH = r"/home/students/student4_8/bert_ner/shopee_data/valid.txt"
        TRAIN_TXT_PATH = r"/home/prodigy/Desktop/bert_ner/shopee_data/valid.txt"
        SAMPLE_PATH  = r"sample.txt"

        read_train_data(TRAIN_TXT_PATH,SAMPLE_PATH)
        os.remove("/home/prodigy/Desktop/bert_ner/shopee_data/valid.txt")
        print('[~]valid.txt removed.')
        os.rename('sample.txt','/home/prodigy/Desktop/bert_ner/shopee_data/valid.txt')
        print('[~]Sample.txt renamed to valid.txt to moved to the shopee_data directory.')

        #Test
        TRAIN_TXT_PATH = r"/home/prodigy/Desktop/bert_ner/shopee_data/test.txt"
        SAMPLE_PATH  = r"sample.txt"

        read_train_data(TRAIN_TXT_PATH,SAMPLE_PATH)
        os.remove("/home/prodigy/Desktop/bert_ner/shopee_data/test.txt")
        print('[~]test.txt removed.')
        os.rename('sample.txt','/home/prodigy/Desktop/bert_ner/shopee_data/test.txt')
        print('[~]Sample.txt renamed to test.txt to moved to the shopee_data directory.')
