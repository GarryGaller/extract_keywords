import os
import re
from pprint import pprint
import extract_keywords
from extract_keywords import (
    preprocessor,
    cleaning, 
    lemmatize, 
    simple_tokenize, 
    textrank,
    tf, idf, stopwords
    )

import gensim
import gensim.summarization as gs
from gensim.summarization import keywords
from nltk.tokenize import sent_tokenize



def test_tfidf(corpus):
    
    corpus = preprocessor(
        corpus,
        include_pos={'NOUN'},  # фильтр по основным граммемам - берем только существительные
        exclude_tag={'Name'},  # фильтр по дополнительным граммемам: игнорируем личные имена
        stopwords=stopwords,   # список stop слов
        ignore_len=3           # длина слов которые будут игнорироваться, если равно или меньше
    )
    
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
        top_n = sorted(
            tokens.items(),
            key=lambda x:x[1],
            reverse=True)[:7] # первые 7 слов по весу TF-IDF
        pprint(top_n)
        
    for idx, tokens in enumerate(result):
        print('type',typ[idx])
        top_n = sorted(
            tokens,
            key=tokens.get,
            reverse=True)[:7] # первые 7 слов по весу TF-IDF
        pprint(top_n)     
        
    
def test_gensim(corpus):    
    '''В gensim используется алгоритм TextRank'''
    
    corpus = preprocessor(
        corpus,
        include_pos={'NOUN'},  # фильтр по основным граммемам - берем только существительные
        exclude_tag={'Name'},  # фильтр по дополнительным граммемам: игнорируем личные имена
        stopwords=stopwords,   # список stop слов
        ignore_len=3           # длина слов которые будут игнорироваться, если равно или меньше
    )
    # так как метод keywords не принимает на обработку ничего, кроме строки текста
    # то делаем всю предварительную работу как обычно, а потом просто конкатенируем
    # через пробел все получившиеся токены
    corpus = [' '.join(tokens)  for tokens in corpus]
    #print(corpus)
    
    for idx,text in enumerate(corpus):
        # по умолчанию scores = False, оценки не выводятся
        top_n = gs.keywords(text,words=7,scores=True) 
        print('type',typ[idx])
        pprint(top_n)


def test_textrank(corpus):
    '''Извлекаем ключевые фразы'''
    
    for idx,text in enumerate(corpus):
        # сегментация на предложения
        sentences = sent_tokenize(text)
        # токенизация\очистка\стемминг или лемматиазция
        # список множеств слов каждого предложения [{a,b,...},{c,d,...},{d,c,a,...},...]
        words = preprocessor(
            sentences,
            include_pos={'NOUN'},  # фильтр по основным граммемам - берем только существительные
            exclude_tag={'Name'},  # фильтр по дополнительным граммемам: игнорируем личные имена 
            uniq=set,              # для этого алгоритма требуется уникализировать слова предложения
            stopwords=stopwords,   # список stop слов
            ignore_len=3           # длина слов которые будут игнорироваться, если равно или меньше
        )
        
        tr, scores, pr = textrank(words,sentences) 
        #pprint(scores)
        #pprint(words)
        # топ ключевых фраз отсортированных по рейтингу
        top_n = tr[:2]
        print('type',typ[idx])
        pprint(top_n)



if __name__ == "__main__":
    import news # тексты для анализа
    typ = ['sport', 'politics',  'economy']
    corpus = [news.sport, news.politics,  news.economy]
    
    test_tfidf(corpus)
    print('-' * 15)
    test_gensim(corpus)
    print('-' * 15)
    test_textrank(corpus)   

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
--------------- 
type sport
[(0,
  0.2726690667680029,
  'Канадский тренер Брайан Орсер, являющийся наставником российской фигуристки '
  'Евгении Медведевой, \n'
  'подтвердил, что не поедет со своей воспитанницей на чемпионат России по '
  'фигурному катанию.'),
 (6,
  0.20719777155135768,
  'Чемпионат России по фигурному катанию пройдёт с 19 по 23 декабря в '
  'Саранске.')]
type politics
[(17, 0.08551061189228006, 'На экспорт в Китай Россия делает большие ставки.'),
 (2,
  0.0829566930899155,
  'Что с этим делать — говорили на первой в истории КНР выставке импортных '
  'товаров.')]
type economy
[(5,
  0.16382613255496373,
  'У городов федерального значения территория равна границам региона, поэтому '
  'на ней запрещено создавать свалки собранного мусора.'),
 (10,
  0.13136988154721627,
  'Тогда часть мусора можно будет перерабатывать на территории городов, что '
  'должно положительно сказаться на стоимости коммунальной услуги по вывозу '
  'мусора.')] 
 
 

''' 
