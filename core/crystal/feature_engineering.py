# coding: utf-8
# Copyright (c) pytmge Development Team.

"""

"""


# import os
# import shutil
# import warnings
import numpy as np

from pytmge.core import _print


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


class feature_engineering:

    @staticmethod
    def feature_selection_by_Pearson_correlation(df_features, c=np.nan, threshold=0.9):
        """
        Feature selection by Pearson correlation.

        The df_features and ds_target should have identical indices.

        Features are removed one by one.
        Each time, find the highest value of the Pearson correlation coefficient
            in the whole correlationmatrix.
        In the two features contributing that highest Pearson correlation coefficient,
            discard the one that having lower coefficient of variance.

        Parameters
        ----------
        df_features : DataFrame
            chemical formulas as index.
        ds_target : Series, optional
            chemical formulas as index.
        threshold : float, optional
            If -1 < threshold < 1, threshold is of the Pearson corrrelation.
            If threshold >= 1, threshold is of the number of selected features (al least n_threshold features will be left)..
            The default is 0.9.

        Returns
        -------
        df_selected_features : DataFrame
            Independent features as index and columns.

        """

        print('\n3-1 feature selection by Pearson correlation ...') if _print else 0
        print('  (this may take a couple of seconds)') if _print else 0

        correlation_matrix = df_features.corr()
        # df_normalized_features = (df_features - np.mean(df_features)) / (np.max(df_features) - np.min(df_features))

        matrix = np.abs(np.round(correlation_matrix * 1, 6))

        # for n in range(len(matrix.index)):
        #     matrix.iloc[n, n] = np.nan
        np.fill_diagonal(matrix.values, np.nan)

        for n in range(len(correlation_matrix.index)):

            a = np.array(matrix)
            m = np.nanmax(a)  # find the maximum in the whole matrix

            if threshold >= 1:
                n_threshold = int(threshold) + 1
                if matrix.shape[1] < n_threshold:
                    break
            elif threshold >= -1:
                p_threshold = threshold
                if m < p_threshold:
                    break
            else:
                break
            i = np.where(a == m)[0][0]
            j = np.where(a == m)[1][0]
            f1 = matrix.columns[i]
            f2 = matrix.columns[j]

            # f1, f2 are the two features having the highest correlation coefficient.
            # Each time, choose one in the two features.

            # Case 1: the feature having lower coefficient of variance is eliminated.
            cv1 = np.std(df_features[f1]) / np.mean(df_features[f1])
            cv2 = np.std(df_features[f2]) / np.mean(df_features[f2])

            # # Case 2: the feature having lower std is eliminated.
            # cv1 = np.std(df_normalized_features[f1])
            # cv2 = np.std(df_normalized_features[f2])

            # # Case 3: the feature having lower correlation with the target is eliminated.
            # cv1 = np.round(np.corrcoef(df_features[f1], ds_target)[0][1], 4)
            # cv2 = np.round(np.corrcoef(df_features[f2], ds_target)[0][1], 4)

            if cv1 > cv2:
                feature_to_drop = f2 * 1
            else:
                feature_to_drop = f1 * 1

            matrix.drop(feature_to_drop, axis=0, inplace=True)
            matrix.drop(feature_to_drop, axis=1, inplace=True)

            # print(n, m, f1, f2, round(cv1, 3), round(cv2, 3))

        df_selected_features = df_features.loc[:, list(matrix.index)] * 1

        print(matrix.shape[1], 'features left.') if _print else 0
        print('max correlation:', m, '\n') if _print else 0

        return df_selected_features
