
JSON_PATH = 'experiments/shopee_data.json'
NEW_WORDS_PATH = 'shopee_data/JSON_files/new_data.json'
FILE = open('shopee_data/JSON_files/new_words.json','w+')
DICT = open('shopee_data/JSON_files/new_dict.json','w+')

MAP = {'FT': 'Fit Type',
       'P' : 'Pattern',
       'S' : 'Sleeves',
       'C' : 'Collar',
       'M' : 'Material',
       }


def tree(shopee_json_data):
    '''
    This functions makess the dictionary simpler
    by removing the second layer dictionary and
    returns a dictionary with attributes as key and
    list of all the words in that attribute as its value

    '''
    dict_, neckline = {}, []
    for key, val in shopee_json_data.items():
        sub_class = []
        for _,val_ in val.items():
            sub_class.extend(val_)
        if key is not 'Neckline': # We do this conditioning because Neckline and collar attributes are merged
            if key == 'Collar':
                dict_.update({'Collar':sub_class.extend(neckline)})
            dict_.update({key:sub_class})
        else:
            neckline = sub_class
    json.dump(dict_, DICT, indent=2)
    return dict_


def finder(shopee_json_data, new_words_data):
    '''
    The function traverses through all the newly
    found words list and selects the words that
    is not in the shopee's json file

    '''

    new_words = []
    for key, val in new_words_data.items():
        tmp = new_words_data[key]
        word = tmp['Word']
        tag = MAP[tmp['Predict']]
        if tag in ('Material', 'Neckline', 'Sleeves','Pattern', 'Fit Type', 'Collar'):
            dict_ = tree(shopee_json_data)
            words = dict_[tag]
            if word not in words:
                new_words.append(tmp)
    return new_words

if __name__ == '__main__' :

    import pandas as pd
    import json
    shopee_json_data = pd.read_json(JSON_PATH)[33] # Reading the shopee's dictionary
    new_words_data = pd.read_json(NEW_WORDS_PATH)['New data'] # Reading the all the words that is considered as newly found

    new_words = finder(shopee_json_data, new_words_data) # Function returns the newly found words after refering with the shopee's dictionary

    json_data = {'New data': new_words}
    json.dump(json_data, FILE, indent=2)
