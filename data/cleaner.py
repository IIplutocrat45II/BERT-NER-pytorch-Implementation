

def cleaner(PATH):
    '''
    This function removes the ' O O O' sample which
    causes the IndexError and AssertionError. Also
    asserts that the labels are the within the VOCAB
    '''
    from tqdm import tqdm
    file_data = open(PATH,'r').read().strip().split('\n\n')
    orig_len = len(file_data)

    sens = []
    for sentence in tqdm(file_data):
        words = sentence.split('\n')
        tmp = []
        for word in words:
            if word != ' O O O':
                tag = word.split()[3]
                if tag in ('FT','O','P','M','C','S'):
                    tmp.append(word)
        local = '\n'.join(tmp)
        sens.append(local)
    data = '\n\n'.join(sens)
    assert len(data.split('\n\n')) == orig_len
    return data

def writer(SAMPLE_PATH, data):
    sample_txt = open(SAMPLE_PATH,'w')
    sample_txt.write(data)


def main():
    import os

    PATH = '/home/prodigy/Desktop/bert_ner/shopee_data/train.txt'
    SAMPLE_PATH  = r"sample.txt"

    writer(SAMPLE_PATH, cleaner(PATH))
    os.rename('sample.txt','/home/prodigy/Desktop/bert_ner/shopee_data/train.txt')

if __name__ == '__main__':
    main()
