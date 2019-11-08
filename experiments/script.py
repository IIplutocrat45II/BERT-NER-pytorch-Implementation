"""semi-supervised vocabulary analysis"""
LOWTHRESH = 0.59
HIGHTHRESH = 0.69
MINFREQ = 20

TXTPATH = r'train.txt' # Change it to the training text file.
DICTPATH = r'shopee.json'
FASTTEXTPATH = r'pretrained_fasttext/cc.en.300.bin'


def preparedata():
    """Get all words in txt and all bottum level values of the dictionary"""

    from collections import defaultdict
    import pandas as pd
    data_dict = defaultdict(int)
    data = pd.read_csv(TXTPATH, sep=' ', header=None)
    dictionary = pd.read_json(DICTPATH)

    for instance in data.iloc[:, 0]:
        data_dict[instance] += 1
    del data
    print("[~]Preparing the data...")
# new_word contain all words in train.txt with enough instances
    new_words = {k: v for k, v in data_dict.items() if v >= MINFREQ}
    del data_dict

    def iter_leafs(vocabulary):
        """
        1. Search the entire nested vocabulary
        2. Returns a generator that contains all bottum level subclasses
        """
        try:
            for _, val in vocabulary.items():
                if isinstance(val, (dict, list)):
                    yield from iter_leafs(val)
                else:
                    yield val
        except (AttributeError, TypeError):
            for val in vocabulary:
                if isinstance(val, (dict, list)):
                    yield from iter_leafs(val)
                else:
                    yield val


# all_subcalsses contain all subclasses
    all_subclasses = []

    for i in range(6):
        all_subclasses.extend(list(iter_leafs(dictionary.iloc[i].values)))
    return new_words, all_subclasses


def main():
    """
    Compare calculated similarity scores with thresholds
    and save the word, the class and the similarity score to respective txt file
    """
    print("[~]Loading the libraries...")
    from gensim.models import fasttext
    from gensim.test.utils import datapath
    fasttextpath = datapath(FASTTEXTPATH)
    model = fasttext.load_facebook_vectors(fasttextpath,
                                           encoding='utf-8')

    new_words, all_subclasses = preparedata()

    def includedigit(word):
        """Detect all word with digits, for fastText doesn't take digits"""
        if isinstance(word, (float, int)):
            return True
        if isinstance(word, str):
            if any(i.isdigit() for i in word):
                return True
        return False


    def duplicate(word):
        """Detect if a word is in dictionary already"""
        if any(word == subclass for subclass in all_subclasses):
            return True
        return False

    print("[~]stripping digits and removing duplicate words...")
    new_words = {k: v for k, v in new_words.items() if not (includedigit(k) or duplicate(k))}



# calculate similarity here
    belowthresholds = 'tmp_files/below_thresholds.txt'
    abovethresholds = 'tmp_files/above_thresholds.txt'
    betweenthresholds = 'tmp_files/between_thresholds.txt'
    import os
    from tqdm import tqdm
    with open(os.path.join(os.getcwd(), belowthresholds), 'w') as file1:
        with open(os.path.join(os.getcwd(), abovethresholds), 'w') as file2:
            with open(os.path.join(os.getcwd(), betweenthresholds), 'w') as file3:
                print("[~]Calculating similarity score and logging the data...")
                for word in tqdm(new_words.keys()):
                    for subclass in all_subclasses:
                        if word == subclass:
                            continue
                        sim_score = model.wv.similarity(word, subclass)
                        if sim_score <= LOWTHRESH:
                            file1.write(f'{word}\n{subclass}\n{sim_score}\n\n')
                        elif sim_score >= HIGHTHRESH:
                            file2.write(f'{word}\n{subclass}\n{sim_score}\n\n')
                        else:
                            file3.write(f'{word}\n{subclass}\n{sim_score}\n\n')


if __name__ == '__main__':
    main()
