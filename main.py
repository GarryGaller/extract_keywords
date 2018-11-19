import os
import re
from pprint import pprint
import extract_keywords
from extract_keywords import (
    clean_text, 
    lemmatize, 
    simple_tokenize, 
    tf, idf, stopwords
    )


if __name__ == "__main__":
    import news # текст для анализа
    
    typ = ['sport', 'politics',  'economy']
    corpus = [news.sport, news.politics,  news.economy]
    
    # примитивная токенизация + нормализация регистра + игнорирование стоп-слов + лемматизация
    corpus = [
                [term for term in lemmatize(
                        clean_text(        # очистка от стоп-слов
                            map(
                                str.lower, # нормализация регистра
                                simple_tokenize(doc) # токенизация
                            ),
                        stopwords,   # список стоп-слов
                        ignore_len=3 # игнорировать слова с длиной меньше или равной 3 букв
                    ),
                   'NOUN',  # берем только существительные 
                   extra={'Name'} # игнорируем имена собственные, фамилии будут использоваться
                   ) 
            ]  for doc in corpus
    ]
    
    
    result = []
    
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
        top= sorted(
            tokens.items(),
            key=lambda x:x[1],
            reverse=True)[:7] # первые 7 слов по весу TF-IDF
        pprint(top)

'''        
type sport
[('чемпионат', 0.20986),
 ('орсера', 0.1574),
 ('медведева', 0.10493),
 ('работа', 0.10493),
 ('катание', 0.10493),
 ('тренер', 0.10493),
 ('россия', 0.1)]
type politics
[('китай', 0.12492),
 ('выставка', 0.09993),
 ('медведев', 0.07495),
 ('товар', 0.07495),
 ('участник', 0.04997),
 ('страна', 0.04997),
 ('продукция', 0.04997)]
type economy
[('мусор', 0.17735),
 ('город', 0.11823),
 ('значение', 0.08867),
 ('год', 0.07918),
 ('регион', 0.05939),
 ('январь', 0.05912),
 ('территория', 0.05912)]
''' 
