import os
import re
from pprint import pprint
import extract_keywords
from extract_keywords import (
    cleaning, 
    lemmatize, 
    simple_tokenize, 
    tf, idf, stopwords
    )

import gensim
import gensim.summarization as gs
from gensim.summarization import keywords


def create_corpus(corpus):
    # примитивная токенизация + нормализация регистра + игнорирование стоп-слов + лемматизация
    corpus = [
                [term for term in lemmatize(
                        cleaning(        # очистка от стоп-слов
                            map(
                                str.lower, # нормализация регистра
                                simple_tokenize(doc) # токенизация
                            ),
                        stopwords,   # список стоп-слов
                        ignore_len=3 # игнорировать слова с длиной меньше или равной 3 букв
                    ),
                   'NOUN',  # берем только существительные 
                   extra={'Name'} # игнорируем личные имена, фамилии будут использоваться
                   ) 
            ]  for doc in corpus
    ]

    return corpus


def test_tfidf(corpus):
    
    corpus = create_corpus(corpus)
    
    result  = []
    
    for tokens in corpus:
        uniq_tokens = set(tokens)
        # создание словаря с весами слов  на основе TF-IDF меры
        ct = {term:
            round(tf(term, tokens) * idf(term, corpus),5)  
            for term in uniq_tokens
        }
        
        result.append(ct) 
     
    for idx, tokens in enumerate(result):
        print('type',typ[idx])
        top = sorted(
            tokens.items(),
            key=lambda x:x[1],
            reverse=True)[:7] # первые 7 слов по весу TF-IDF
        pprint(top)
        
    for idx, tokens in enumerate(result):
        print('type',typ[idx])
        top = sorted(
            tokens,
            key=tokens.get,
            reverse=True)[:7] # первые 7 слов по весу TF-IDF
        pprint(top)     
        
    
def test_gensim(corpus):    
    '''В gensim используется алгоритм TextRank'''
    
    corpus = create_corpus(corpus)
    # так как метод keywords не принимает на обработку ничего, кроме строки текста
    # то делаем всю предварительную работу как обычно, а потом просто конкатенируем
    # через пробел все получившиеся токены
    corpus = [' '.join(tokens)  for tokens in corpus]
    #print(corpus)
    
    for idx,text in enumerate(corpus):
        # по умолчанию scores = False, оценки не выводятся
        res = gs.keywords(text,words=7,scores=True) 
        print('type',typ[idx])
        pprint(res)



if __name__ == "__main__":
    import news # тексты для анализа
    typ = ['sport', 'politics',  'economy']
    corpus = [news.sport, news.politics,  news.economy]
    
    test_tfidf(corpus)
    print('-' * 15)
    test_gensim(corpus)
        

'''        
type sport
[('чемпионат', 0.20986),
 ('орсера', 0.1574),
 ('катание', 0.10493),
 ('работа', 0.10493),
 ('медведева', 0.10493),
 ('тренер', 0.10493),
 ('россия', 0.1)]
type politics
[('китай', 0.12492),
 ('выставка', 0.09993),
 ('товар', 0.07495),
 ('медведев', 0.07495),
 ('участник', 0.04997),
 ('продукция', 0.04997),
 ('страна', 0.04997)]
type economy
[('мусор', 0.17735),
 ('город', 0.11823),
 ('значение', 0.08867),
 ('год', 0.07918),
 ('регион', 0.05939),
 ('законопроект', 0.05912),
 ('территория', 0.05912)]

type sport
['чемпионат', 'орсера', 'катание', 'работа', 'медведева', 'тренер', 'россия']
type politics
['китай', 'выставка', 'товар', 'медведев', 'участник', 'продукция', 'страна']
type economy
['мусор', 'город', 'значение', 'год', 'регион', 'законопроект', 'территория']
---------------
type sport
[('чемпионат россия катание', 0.3068327671271174),
 ('тренер орсера', 0.27558741871717746),
 ('работа', 0.25563835229876686),
 ('медведева', 0.19478566973449646)]
type politics
[('выставка товар', 0.2805391712616916),
 ('медведев', 0.2447357455937165),
 ('китаи участник', 0.2267156724927703),
 ('россия продукция', 0.17115119843682902)]
type economy
[('мусор', 0.3911450765082161),
 ('год регион', 0.28148914010665266),
 ('город', 0.23050017112235235),
 ('мегаполис', 0.20028980231724147),
 ('отход', 0.19137889757997345),
 ('утилизация', 0.18616344597515197)]

''' 
