import pandas as pd
from sklearn.model_selection import train_test_split


def banking(data = 'banking_data.csv'):
    
    df = pd.read_csv(data)

    numerical = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(exclude='number').columns.tolist()
    to_encode = ('default', 'housing', 'loan', 'y')
    
    categorical = [col for col in categorical if col not in to_encode]

    for cat in exclude:
        df[f'{cat}_encoded'] = df[cat].map({'no':0,'yes':1})
    
    dummies_df = pd.get_dummies(df[['job','marital','education', 'contact','month','poutcome']], drop_first=False, dtype='int')

    df = pd.concat([df,df_dummy], axis = 1)


    return df


def data_split(df, col):

    random_seed = 1729
    
    train, test= train_test_split(df, test_size= .3, random_state = seed, stratify= df[col])
    validate, test = train_test_split(test, test_size = 0.5, random_state = seed, stratify = test[col])

    return train, validate, test

    