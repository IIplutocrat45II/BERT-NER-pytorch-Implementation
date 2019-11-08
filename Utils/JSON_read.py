import json

def read_data(path='/home/prodigy/Desktop/bert_ner/shopee_data/JSON_files/test_data.json'):
    print('[~]Reading the JSON file...')

    with open(path,'r') as f:
        data = f.read()
    file = open('/home/prodigy/Desktop/bert_ner/shopee_data/JSON_files/new_data.json','w+')

    obj = json.loads(data)
    new = []
    new_words = []

    print('[~]Finding the new found words...')
    for i, dic in enumerate(obj['Test data']):

        words = dic['Text'].strip().split()
        tags = dic['Tags'].strip().split()
        predict = dic['predict'].strip().split()
        for w, t, p in zip(words, tags, predict):
            if (w in new_words) == True:
                continue
            else:
                new_words.append(w)
            if t != p and p !='O' and t == 'O':
                tmp = {'Line number': i,
                       'Word': w,
                       'Tags': t,
                       'Predict': p,
                       'Text': words,
                       'tags' : tags
                       }
                #print(tmp,'\n')
                new.append(tmp)

    json_data = {'New data': new}
    json.dump(json_data, file, indent=2)
    print('[~]Data written into new_data.json')
    file.close()

if __name__ == '__main__':
    read_data()
