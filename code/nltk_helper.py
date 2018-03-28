import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem.porter import PorterStemmer

def word_data_preprocessing(df):
    
    try:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient.split(" "):
                    word.append(WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', pos)))
                instance.append(' '.join(word))
            instances.append(instance)
        
        df['ingredients_string_word'] = instances
        df['ingredients_string_word'] = [','.join(i).strip().lower() for i in df['ingredients_string_word']]  
        
        x_data = df['ingredients_string_word'].tolist()
        y_data = df['cuisine'].tolist()
        
        return x_data, y_data
    
    except:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient.split(" "):
                    word.append(WordNetLemmatizer().lemmatize(pos))
                instance.append(' '.join(word))
            instances.append(instance)
        
        df['ingredients_string_word'] = instances
        df['ingredients_string_word'] = [','.join(i).strip().lower() for i in df['ingredients_string_word']]  
        x_data = df['ingredients_string_word'].tolist()

        return x_data, None
    
def char_data_preprocessing(df):
    
    try:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient:
                    instance.append(WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', pos)))
            instances.append(''.join(instance))
        
        df['ingredients_string_char'] = instances
        
        x_data = df['ingredients_string_char'].tolist()
        y_data = df['cuisine'].tolist()
        
        return x_data, y_data
    
    except:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient.split(" "):
                    instance.append(WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', pos)))
            instances.append(''.join(instance))
        
        df['ingredients_string_char'] = instances
        x_data = df['ingredients_string_char'].tolist()

        return x_data, None
    
def token_data_preprocessing(df):
    
    try:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient.split(" "):
                    instance.append(WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', pos)))
            instances.append(instance)
        
        df['ingredients_string_token'] = instances
        df['ingredients_string_token'] = [','.join(i).strip().lower() for i in df['ingredients_string_token']]  
        
        x_data = df['ingredients_string_token'].tolist()
        y_data = df['cuisine'].tolist()
        
        return x_data, y_data
    
    except:
        instances = []
        for ingredients in df['ingredients']:
            instance = []
            for ingredient in ingredients:
                word = []
                for pos in ingredient.split(" "):
                    instance.append(WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', pos)))
            instances.append(instance)
        
        df['ingredients_string_token'] = instances
        df['ingredients_string_token'] = [','.join(i).strip().lower() for i in df['ingredients_string_token']]  
        x_data = df['ingredients_string_token'].tolist()

        return x_data, None
    
