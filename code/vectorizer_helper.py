from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk

def tfidf_vectorizer(x_train, x_test, analyzer):
    
    tfidf = TfidfVectorizer(analyzer=lambda d: d.split(analyzer)).fit(x_train)
    x_train = tfidf.fit_transform(x_train).toarray()
    x_test = tfidf.transform(x_test).toarray()
    
    return x_train, x_test, tfidf

def char_vectorizer(x_train, x_test, analyzer):
            
    countvec = CountVectorizer(analyzer=analyzer).fit(x_train)
    x_train = countvec.fit_transform(x_train).toarray()
    x_test = countvec.transform(x_test).toarray()
    
    return x_train, x_test, countvec

def jump_value(df, column, model, data, x_train, x_test, jump_number, total_count, label_count):
    
    total_dictionary = [i.split(",") for i in data]
    total_dictionary = [j for i in total_dictionary for j in i]
    for label in column:
        label_dictionary = [data[i].split(",") for i in range(len(data)) if df['cuisine'][i] == label]
        label_dictionary = [j for i in label_dictionary for j in i]

        total_fdist = nltk.FreqDist(total_dictionary)
        label_fdist = nltk.FreqDist(label_dictionary)

        total_column = [value[0] for value in total_fdist.most_common(total_count)]
        label_column = [value[0] for value in label_fdist.most_common(label_count)]
    
        differs = set(label_column) - set(total_column)
        for differ in differs:
            x_train[:,model.vocabulary_[differ]] = x_train[:,model.vocabulary_[differ]] * jump_number
            x_test[:,model.vocabulary_[differ]] = x_test[:,model.vocabulary_[differ]] * jump_number
    return x_train, x_test
    
    
    
    
    