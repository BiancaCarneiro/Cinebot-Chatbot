import torch
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # pip3 install wordcloud


def prepare_sequence(seq: list, word_to_ix:dict, device:torch.DeviceObjType):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def one_hot_encoding_mapper(data:list) -> tuple[dict, dict]:
    word_to_ix = {}
    tag_to_ix = {}
    for sentence, tag in data:
        for idx in range(len(sentence)):
            if sentence[idx] not in word_to_ix:
                word_to_ix[sentence[idx]] = len(word_to_ix)
            if tag[idx] not in tag_to_ix:
                tag_to_ix[tag[idx]] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    return input_string.translate(translator)

def plot_hist(data:list, title:str="") -> None:
    plt.hist(data)
    plt.title(title)
    plt.grid()
    plt.show()
    
def plot_most_frequent_word(tokenized_words:list, stopwords:list=[]) -> None:
    counter=Counter(tokenized_words)
    most=counter.most_common()

    x, y= [], []
    for word,count in most[:40]:
        if (word not in stopwords):
            x.append(word)
            y.append(count)

    sns.barplot(x=y,y=x)
    
def cloud_of_words(text: str, stopwords: list, filename:str=None) -> None:
    wordcloud = WordCloud(max_font_size=60, max_words=150, background_color="white", collocations=False, stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    if filename:
        wordcloud.to_file(filename)
        
        
def modify_dataset(data:list) -> list:
    ner_df = []

    for sentence in data:
        words_lists = []
        entities = []
        last_index = 0
        for entity in sentence[1]["entities"]:
            if last_index < entity[0]:
                list_words = sentence[0][last_index: entity[0]].split()
                words_lists += list_words
                entities += ["O"]*len(list_words)
                
            list_words = sentence[0][entity[0]: entity[1]].split()
            words_lists += list_words
            entities += [entity[2]]*len(list_words)
            last_index = entity[1]
            
        ner_df.append((
            words_lists,
            entities
        ))
    
    return ner_df