import pandas as pd
import jieba

def readFile(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    #去除行末的换行符，否则会在停用词匹配的过程中产生干扰
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines

def wordCut (lines):
    stopwords = readFile("stopwords.txt")
    userStopwords = ['\t', ' ', ',', '~', '_', '...', '…', '##', '|', '!', '🤣', '❤', '🙏', '\u200b', 'http', 't', 'cn', '☎', '❌', '❗', '〜']
    stopwords += userStopwords
    wordLists = []
    for line in lines :
        wordList = []
        wordList += [word for word in jieba.cut(line) if word not in stopwords]
        wordLists.append(' '.join(wordList))
    return wordLists

def preprocess(dataset):
    # dataset = pd.read_csv('train.csv')
    dataset.fillna (' ', inplace = True)
    dataset['text'] = dataset['content'] + dataset['comment_all']
    dataset.drop (dataset.columns[[0, 1, 2]], axis = 1, inplace = True)
    dataset["textCut"] = wordCut(dataset["text"])
    print (dataset)
    return dataset

dataset = pd.read_csv('train.csv')
preprocess(dataset).to_csv ("trainCut.csv")
dataset = pd.read_csv('test.csv')
preprocess(dataset).to_csv ("testCut.csv")