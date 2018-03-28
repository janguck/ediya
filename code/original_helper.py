import pandas as pd

def data_preprocessing(df):
    
    try:
        df['ingredients_string'] = [','.join(i).strip().lower() for i in df['ingredients']]  
        x_data = df['ingredients_string'].tolist()
        y_data = df['cuisine'].tolist()
        
        return x_data, y_data
    
    except:
        df['ingredients_string'] = [','.join(i).strip().lower() for i in df['ingredients']]  
        x_data = df['ingredients_string'].tolist()

        return x_data, None
        
