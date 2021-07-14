import json
from transformers import BertTokenizer
from transformers import AutoTokenizer
from gensim.models import Word2Vec

file_name ='.\\dureader_raw\\raw\\trainset\\search.train.json'


def main():
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
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

    f = open(file_name, 'r', encoding='utf-8')
    for lines in f:
        obj = json.loads(lines.strip())
        for val in obj['documents']:
            title = val['title']
            statistic_tokenizer(title)
            paragraphs = val['paragraphs']
            for paragraph in paragraphs:
                statistic_tokenizer(paragraph)
    f.close()
    statistic_val=sorted(statistic.items(), key=lambda d:d[1], reverse=True)
    print(statistic_val)
    size = len(statistic.items())

    model = Word2Vec(statistic.keys(), vector_size=size)
    model.wv.save_word2vec_format('.\model_vec.txt', fvocab=None, binary=False)

if __name__ == '__main__':
    main()
