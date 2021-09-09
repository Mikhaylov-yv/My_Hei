import data_preparation as dp
import pandas as pd
import numpy as np


def open_data(path):
    df = pd.read_csv(path, sep=';')
    df['InvoiceMonth'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('M')
    df['InvoiceTime'] = pd.to_datetime(df['InvoiceDate']).dt.hour
    df['InvoiceDate_Day'] = pd.to_datetime(df['InvoiceDate']).dt.date
    df['Sum'] = df['PricePerItem'] * df['Quantity']
    return df

df_in = open_data('data/train_data.csv')
month_list = np.sort(df_in['InvoiceMonth'].unique())

df_clean = df_in
# Удаляем данные без указания клиента
df_clean = df_clean.dropna(subset=['CustomerID'])

def test_d_prep_all():
    dt = dp.Data_preparation(df_clean,month_list)
    X_train, y_test, X_test, y_test = dt.preparation()
    print(
        'Получено: ', dt.df_Cust.shape[1], '\n',
          'Должно быть: ', dt.col_gr * len(dt.month_list)
          )
    print(X_train.shape, y_test.shape, X_test.shape, y_test.shape)

def test_d_prep_2():
    df_test = df_in[df_in.CustomerID.isin([16919.0, 16843.0])]
    # df_test = df_test.loc[df_test.InvoiceMonth < '2011-12']
    test_data = dp.Data_preparation(df_test,month_list, 'last', 'pred').preparation()
    print(test_data.shape)