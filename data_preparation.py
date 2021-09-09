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
        self.col_gr = 5
        # Периоды для средней
        self.n_min = 2
        self.n_max = 4
        # Временной интервал для предсказания
        self.inter = 10
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
            X = ar[-self.df_Cust.shape[0]:,
                -1 * self.col_gr * (self.inter -1): ]
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
        n_min = self.n_min
        n_max = self.n_max
        for i, month in enumerate(month_list):
            df_Cust[(month, 'Sum_buy')] = df_in.loc[(df_in['Quantity'] > 0)
                                                 &
                                                 (df_in['InvoiceMonth'] == month),
                                                 ['CustomerID', 'Sum']].groupby('CustomerID').sum()
        df_Cust_ = df_Cust

        for i, month in enumerate(month_list):
            df_Cust[(month, 'y_buy')] = (df_in.loc[(df_in['InvoiceMonth'] == month)
                                                &
                                                (df_in['Sum'] > 0),  # Исключаем возвраты товаров
                                                'CustomerID'].value_counts() > 0).astype(int)
            df_Cust[(month, 'Sum_return')] = df_in.loc[(df_in['Quantity'] < 0)
                                                    &
                                                    (df_in['InvoiceMonth'] == month),
                                                    ['CustomerID', 'Sum']].groupby('CustomerID').sum()

            df_Cust[(month, 'mov_av_sum')] = df_Cust_.loc[:, month_list[i - n_min: i + 1]] \
                                                 .mean(1) - df_Cust_.loc[:,month_list[
                                                                         i - n_max: i + 1]].mean(1)
        # print(df_Cust.shape)
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
        # print(n_0, count_per)
        iter_list = list(range(self.n_0 * self.col_gr,
                               df_Cust.shape[1] - (self.col_gr * inter - 1),
                               self.col_gr))
        # print(df_Cust.shape)
        # print(iter_list)
        for i in iter_list:
            i_f = i
            i_l = i + self.col_gr * inter
            # print(i_f, i_l)
            ar = df_Cust.values[:, i_f:i_l]
            # print(ar.shape)
            if i > 0:
                ar_all = np.vstack((ar_all, ar))
            else:
                ar_all = ar
            if self.period in ['first', 'last']: break
        return ar_all
        #
        # if self.goal == 'learn':
        #     X_train, y_train = ar_train[:, :-4], ar_train[:, -1]
        #     X_test, y_test = ar_test[:, :-4], ar_test[:, -1]
        #     return X_train, y_train, X_test, y_test, count_per
        # if self.goal == 'pred':
        #     return ar_all[:, 4:]
