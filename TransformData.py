class TransformData:
    pairs = [2 * i for i in range(25)]
    impairs = [2 * i + 1 for i in range(25)]

    @classmethod
    def transform_data(cls, df):

        df['sum_50'] = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].sum(axis=1)
        df['variance_50'] = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].var(axis=1)
        df['mean_50'] = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].mean(axis=1)

        df['sum_diff'] = cls._sum_diff(df)
        df['is_pair'] = cls._is_pair(df)
        df['is_impair'] = cls._is_impair(df)
        df['is_under_10'] = cls._is_under(df, 10)
        df['is_under_20'] = cls._is_under(df, 20)
        df['boule_1_freq'] = cls._freq_val(df, 'boule_1')
        df['boule_2_freq'] = cls._freq_val(df, 'boule_2')
        df['boule_3_freq'] = cls._freq_val(df, 'boule_3')
        df['boule_4_freq'] = cls._freq_val(df, 'boule_4')
        df['boule_5_freq'] = cls._freq_val(df, 'boule_5')

        df['etoile_1_freq'] = cls._freq_val(df, 'etoile_1')
        df['etoile_2_freq'] = cls._freq_val(df, 'etoile_2')
        df['is_pair_etoile'] = cls._is_pair_etoile(df)
        df['is_impair_etoile'] = cls._is_impair_etoile(df)

        return df

    @staticmethod
    def _is_under(df, number):
        return ((df['boule_1'] <= number).astype(int) +
                (df['boule_2'] <= number).astype(int) +
                (df['boule_3'] <= number).astype(int) +
                (df['boule_4'] <= number).astype(int) +
                (df['boule_5'] <= number).astype(int))

    @staticmethod
    def _is_pair(df):
        return ((df['boule_1'].isin(TransformData.pairs)).astype(int) +
                (df['boule_2'].isin(TransformData.pairs)).astype(int) +
                (df['boule_3'].isin(TransformData.pairs)).astype(int) +
                (df['boule_4'].isin(TransformData.pairs)).astype(int) +
                (df['boule_5'].isin(TransformData.pairs)).astype(int))

    @staticmethod
    def _is_impair(df):
        return ((df['boule_1'].isin(TransformData.impairs)).astype(int) +
                (df['boule_2'].isin(TransformData.impairs)).astype(int) +
                (df['boule_3'].isin(TransformData.impairs)).astype(int) +
                (df['boule_4'].isin(TransformData.impairs)).astype(int) +
                (df['boule_5'].isin(TransformData.impairs)).astype(int))

    @staticmethod
    def _is_pair_etoile(df):
        return ((df['etoile_1'].isin(TransformData.pairs)).astype(int) +
                (df['etoile_2'].isin(TransformData.pairs)).astype(int))

    @staticmethod
    def _is_impair_etoile(df):
        return ((df['etoile_1'].isin(TransformData.impairs)).astype(int) +
                (df['etoile_2'].isin(TransformData.impairs)).astype(int))

    @staticmethod
    def _sum_diff(df):
        return ((df['boule_2'] - df['boule_1']) ** 2 +
                (df['boule_3'] - df['boule_2']) ** 2 +
                (df['boule_4'] - df['boule_3']) ** 2 +
                (df['boule_5'] - df['boule_4']) ** 2)

    @staticmethod
    def _freq_val(df, column):
        tab = df[column].values.tolist()
        freqs = []
        pos = 1
        for e in tab:
            freqs.append(tab[0:pos].count(e))
            pos = pos + 1
        return freqs




