import math
 


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


def idf(term, corpus, smooth_idf=None):
    """
    idf - логарифм от количества текстов в корпусе, 
    разделенное на число текстов, где термин встречается. 
    Если термин не появляется в корпусе или появляется во всех документах, 
    возвращается 0.0.
    Принимает слово, для которого считаем IDF и корпус документов 
    в виде списка списков слов.
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
 



if __name__ == "__main__":
    pass
