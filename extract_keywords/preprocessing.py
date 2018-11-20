from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
from .stopwords import stopwords        
import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()


def nltk_preprocessor(sentences):
    ''' токенизация + стемминг'''
    
    tokenizer = RegexpTokenizer(r'\w+')
    # стемминг до корневой основы
    lmtzr = RussianStemmer()
    words = [set(lmtzr.stem(word)                                # стемминг
                for word in tokenizer.tokenize(sentence.lower()) # токенизация
             )
             for sentence in sentences
    ]
    return words


def preprocessor(corpus, 
        include_pos={'NOUN'},
        exclude_tag={'Name'}, 
        uniq=list):
    '''токенизация + 
    нормализация регистра + 
    игнорирование стоп-слов +
    лемматизация'''
    
    words = [uniq(
                lemmatize(                                 # лемматизация 
                    cleaning(                              # очистка от стоп-слов
                        simple_tokenize(text.lower()),     # токенизация
                        stopwords,                         # список стоп-слов
                        ignore_len=3                       # игнорировать слова с длиной меньше или равной 3 символам
                    ),
                   include_pos=include_pos,                # фильтр по основным граммемам 
                   exclude_tag=exclude_tag                 # фильтр по дополнительным граммемам 
                )
        )  for text in corpus      
    ]
    
    return words
    
 
def cleaning(tokens,stopwords,ignore_len=0):
    '''Очистка текста он слов не имеющих смысловой важности в документе'''
    for term in tokens:
        if term in stopwords or len(term) <= ignore_len:
            continue        
        yield term 
        
        
def simple_tokenize(text):
    return re.findall('\w+',text)


def lemmatize(tokens, include_pos={'NOUN'}, exclude_tag=None):
    '''Возвращает из переданного списка только те слова, 
    чья часть речи совпадает с указанной граммемой OpenCorpora
    http://opencorpora.org/dict.php?act=gram. 
    Слово преобразуется к нормальной форме'''
    
    for term in tokens:
        # первый объект Parse из всех возможных грамматических разборов слова
        parse = morph.parse(term)[0] 
        tag = parse.tag  # OpencorporaTag
        # если слово не заданная часть речи или в нем есть характеристики,
        # по которым мы хотим фильтровать слова
        if (include_pos and not (tag.POS in include_pos)) or (exclude_tag in tag):
            continue
        else:
            # приводим женские фамилии с окончанием на -ой 
            # (род., дат., творит., предл. падежи ) к правильной форме
            if {'femn','Surn'} in tag:
                # склоняем в именительный падеж един. число женского рода
                nf = parse.inflect({'nomn', 'sing' ,'femn'}).word
            else:
                # у прочих слов просто получаем нормальную форму: 
                # имен. падеж ед. число
                nf = parse.normal_form
            yield nf              
    