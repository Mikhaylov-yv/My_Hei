import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Data_preparation:

    def __init__(self, df, month_list,
                 period='all',  # all, first, last
                 goal='learn'  # learn, pred
                 ):
        self.df = df
        self.month_list = np.sort(month_list)
        self.period = period
        self.goal = goal
        self.n_0 = 0
        self.col_gr = 1
        # Временной интервал для предсказания
        self.inter = 18
        if period == 'last':
            self.n_0 = len(self.month_list) - (self.col_gr * self.inter)

    # Оновная функция

    def preparation(self):
        df_in = self.df
        self.df_Cust = self.retrieval_data(df_in)
        self.Cust_list = self.df_Cust.index
        if self.goal == 'learn':
            train, test = self.get_train_test(self.df_Cust)
            train = self.get_periods(train)
            test = self.get_periods(test)
            X_train, y_train = train[:, :-self.col_gr], train[:, -1]
            X_test, y_test= test[:, :-self.col_gr], test[:, -1]
            return X_train, y_train, X_test, y_test
        if self.goal == 'pred':
            ar = self.get_periods(self.df_Cust)
            # print(ar.shape)
            # print(-1 * self.col_gr * (self.inter))
            X = ar[-self.df_Cust.shape[0]:,
                -1 * self.col_gr * (self.inter): ]
            return X


    def get_train_test(self, df):
        return train_test_split(df, test_size=0.15, random_state=17)

    # Функция извлечения признаков
    def retrieval_data(self, df_in):
        month_list = self.month_list
        customer_list = df_in['CustomerID'].unique()
        print('Клиентов:', len(customer_list))
        df_Cust = pd.DataFrame(index=customer_list,
                               columns=pd.MultiIndex.from_tuples([(month, 'buy') for month in month_list]),
                               )


        for i, month in enumerate(month_list):
            df_Cust[(month, 'buy')] = (df_in.loc[(df_in['InvoiceMonth'] == month)
                                                &
                                                (df_in['Sum'] > 0),  # Исключаем возвраты товаров
                                                'CustomerID'].value_counts() > 0).astype(int)
        df_out = df_Cust.sort_index(axis=1)
        df_out = df_out.fillna(0)
        return df_out

    # Функция разделения набора данных по периодами

    def get_periods(self, df_Cust):
        month_list = self.month_list
        inter = self.inter
        col_gr = self.col_gr
        count_month = len(month_list)
        count_per = count_month - inter
        n_0 = self.n_0
        # print(df_Cust.shape)
        iter_list = list(range(self.n_0 * col_gr,
                               df_Cust.shape[1] - (col_gr * inter - 1),
                               col_gr))
        # print(df_Cust.shape)
        # print(iter_list)
        for i_, i in enumerate(iter_list):
            i_f = i
            i_l = i + col_gr * inter
            # print(i_f, i_l)
            ar = df_Cust.values[:, i_f:i_l]
            if i_ > 0:
                # print(ar.shape, ar_all.shape)
                ar_all = np.vstack((ar_all, ar))
            else:
                ar_all = ar
            if self.period in ['first', 'last']: break
        #     Добавим столбец количества последних позитивных месяцев
        # print(ar_all.shape)
        if self.goal == 'pred':
            ar_ = self.get_count_last_pz_periods(ar_all)
        else:
            ar_ = self.get_count_last_pz_periods(ar_all[:, :-1])
        ar_all = np.concatenate((ar_, ar_all), axis=1)
        return ar_all


    def get_count_last_pz_periods(self, ar):
        ar_out = np.zeros((ar.shape[0], 1))
        for i in range(ar.shape[1]):
            ar_out[:, 0] += ar[:, i]
            ar_out[:, 0] *= ar[:, i]
        return ar_out
