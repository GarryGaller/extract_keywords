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
        include_pos={'NOUN','ADJF'},  # фильтр по основным граммемам - берем только существительные и прилагательные
        exclude_tag={'Name'},         # фильтр по дополнительным граммемам: игнорируем личные имена
        stopwords=stopwords,          # список stop слов
        ignore_len=3                  # длина слов которые будут игнорироваться, если равно или меньше
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
        include_pos={'NOUN','ADJF'},  # фильтр по основным граммемам - берем только существительные и прилагательные
        exclude_tag={'Name'},         # фильтр по дополнительным граммемам: игнорируем личные имена
        stopwords=stopwords,          # список stop слов
        ignore_len=3                  # длина слов которые будут игнорироваться, если равно или меньше
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
[('чемпионат', 0.1646),
 ('орсера', 0.12345),
 ('тренер', 0.0823),
 ('катание', 0.0823),
 ('медведева', 0.0823),
 ('фигурный', 0.0823),
 ('работа', 0.0823)]
type politics
[('китай', 0.09124),
 ('выставка', 0.073),
 ('медведев', 0.05475),
 ('товар', 0.05475),
 ('российский', 0.03666),
 ('продукция', 0.0365),
 ('участник', 0.0365)]
type economy
[('мусор', 0.14148),
 ('город', 0.09432),
 ('значение', 0.07074),
 ('коммунальный', 0.07074),
 ('федеральный', 0.07074),
 ('год', 0.06317),
 ('регион', 0.04738)]

type sport
['чемпионат', 'орсера', 'тренер', 'катание', 'медведева', 'фигурный', 'работа']
type politics
['китай','выставка','медведев','товар','российский','продукция','участник']
type economy
['мусор', 'город', 'значение', 'коммунальный', 'федеральный', 'год', 'регион']
---------------
type sport
[('чемпионат россия', array([0.3367464])),
 ('тренер орсера', array([0.28151527])),
 ('работа', array([0.23236174])),
 ('медведева', array([0.17928001])),
 ('катание', array([0.17357796]))]
type politics
[('выставка', array([0.27940882])),
 ('китаи', array([0.23904963])),
 ('медведев', array([0.21263316])),
 ('россиискии', array([0.21183336])),
 ('товар', array([0.20526634])),
 ('китаискии', array([0.15919711])),
 ('национальныи', array([0.15329746]))]
type economy
[('мусор', array([0.36875813])),
 ('год регион', array([0.26281802])),
 ('город', array([0.21854621])),
 ('мегаполис', array([0.18352752])),
 ('законопроект', array([0.17506299])),
 ('утилизация', array([0.16926709]))]
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
