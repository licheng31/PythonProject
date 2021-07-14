import json
from transformers import BertTokenizer
from transformers import AutoTokenizer
from gensim.models import Word2Vec

file_name = '.\\squad_data\\train-v2.0.json'

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    f = open(file_name, mode='r', encoding='utf-8')
    statistic = {}

    def statistic_tokenizer(str):
        if len(str) > 512:
            str = str[:512]
        lst = tokenizer.tokenize(str)
        for words in lst:
            if words in statistic.keys():
                statistic[words] += 1
            else:
                statistic[words] = 1

    json_val = json.load(f)
    datas = json_val['data']
    for data in datas:
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                statistic_tokenizer(question)
                for answer in qa['answers']:
                    text = answer['text']
                    statistic_tokenizer(text)
    f.close()
    items = list(statistic.items())
    #print(items)
    statistic_val = sorted(items, key=lambda item: item[1],reverse=True)

    print(statistic_val)

if __name__ == '__main__':
    main()