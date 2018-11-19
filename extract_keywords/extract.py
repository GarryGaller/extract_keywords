import math
import pymorphy2
import re
morph = pymorphy2.MorphAnalyzer()


def simple_tokenize(text):
    return re.findall('\w+',text)


def lemmatize(tokens,grammeme=None, extra=None):
    '''Возвращает из переданного списка только те слова, 
    чья часть речи совпадает с указанной граммемой OpenCorpora
    http://opencorpora.org/dict.php?act=gram. 
    Слово преобразуется к нормальной форме'''
    
    for term in tokens:
        # первый объект Parse из всех возможных грамматических разборов слова
        parse = morph.parse(term)[0] 
        tag = parse.tag  # OpencorporaTag
        # если слово == часть_речи_существительное
        if grammeme and tag.POS != grammeme or extra in tag:
            continue
        else:
            # приводим женские фамилии с окончанием на -ой 
            # (род., дат., творит.падежи ) к правильной форме
            if {'femn','Surn'} in tag:
                # склоняем в именительный падеж един. число женского рода
                nf = parse.inflect({'nomn','sing' ,'femn'}).word
            else:
                # у прочих слов просто получаем нормальную форму: имен. падеж ед. число
                nf = parse.normal_form
            yield nf    


def tf(term, words, sublinear_tf=False):
    """
    TF термина - количество раз, когда термин встретился в тексте 
    разделенное на количество всех слов в тексте
    The frequency of the term in text. 
    """
    _tf = words.count(term) / len(words)
    
    if sublinear_tf:
        _tf = math.log(_tf) + 1
   
    return _tf 


def idf(term, corpus,smooth_idf=None):
    """
    idf - логарифм от количества текстов в корпусе, 
    разделенное на число текстов, где термин встречается. 
    Если термин не появляется в корпусе или появляется во всех документах, возвращается 0.0.
    Принимает слово, для которого считаем IDF и корпус документов в виде списка списков слов.
    """
    n_samples = len(corpus)
    
    #cnt = sum(1.0 for words in corpus if words & {term})  or 1.0  
    df = sum(1.0 for tokens in corpus if term in tokens)  or 1.0 
    
    if smooth_idf:
        df += int(smooth_idf)
        n_samples += int(smooth_idf)
        _idf = math.log(n_samples / df ) + 1  
        # формула из sklearn.feature_extraction.text
    else:
        _idf = math.log(n_samples / df ) + 1   # len(corpus) / cnt + 1 # standard textbook notation 
    
    return _idf  
 

def clean_text(tokens,stopwords,ignore_len=2047):
    for term in tokens:
        if term in stopwords or len(term) <= ignore_len:
            continue        
        yield term



    
        
            
       
        
         
    
    



         
