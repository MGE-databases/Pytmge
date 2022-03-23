# coding: utf-8
# Copyright (c) pytmge Development Team.

"""
Extracting features based on electron orbital attributes.

"""


import os
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from pytmge.core import elemental_data
from pytmge.core import progressbar, _print


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


class feature_design:
    '''
    Extracting features based on electron orbital attributes.

    '''

    def __init__(self):
        self._elemental_attribute_format = '[attribute].[shell_selection].[math operator 1]'
        self._feature_format = '[attribute].[shell_selection].[math operator 1].[math operator 2]'

    @staticmethod
    def delete_unusable_features(df_features):
        '''
        Delete the features having empty value(s)
        and the features being of zero variance.

        Parameters
        ----------
        df_features : DataFrame
            features.

        Returns
        -------
        df_usable_features : DataFrame
            usable_features.

        '''

        print('deleting unusable features') if _print else 0

        features_variance = np.var(df_features, axis=0)
        df_usable_features = df_features.loc[:, features_variance != 0] * 1

        na = list(df_usable_features.isna().any())
        na = [False if i else True for i in na]
        df_usable_features = df_usable_features.loc[:, na] * 1

        # df_usable_features.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'usable_feature_variables.csv')

        print(df_usable_features.shape[0], 'entries', df_usable_features.shape[1], 'usable features') if _print else 0

        return df_usable_features

    @classmethod
    def get_features(self, df_composition):
        '''
        Extracting features.

        Parameters
        ----------
        df_composition : DataFrame
            df_composition.

        Returns
        -------
        df_features : DataFrame
            features.

        '''

        print('\n  calculating features ...') if _print else 0

        dict_oa = elemental_data().orbital_attributes_of_elements
        df_orbital_attributes_of_elements = pd.DataFrame.from_dict(dict_oa, orient='index')

        # # Lite edition
        # _ea = df_orbital_attributes_of_elements.loc[
        #     [
        #         'E' in col_name
        #         and 'range' in col_name
        #         for col_name in list(df_orbital_attributes_of_elements.index)
        #     ], :
        # ]
        # df_orbital_attributes_of_elements = _ea * 1
        # #

        print(df_orbital_attributes_of_elements.shape[0], 'attributes,', df_composition.shape[0], 'entries.')

        cache_path = str(Path(__file__).absolute().parent) + '\\' + '_cache\\'
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.makedirs(cache_path)

        chemical_formula_list = list(df_composition.index)
        elements_existence = (df_composition.notnull() * 1).replace(0, np.nan)

        math_operators = {
            'sum': 'sum(x)',
            'avg': 'sum(x)/N',
            'wavg': 'sum(w*x)/sum(w)',
            'max': 'max(x)',
            'min': 'min(x)',
            'range': 'max(x)-min(x)',
            'std': '(sum((x-wavg)**2)/N)**(1/2)'
        }

        warnings.filterwarnings('ignore')

        i = 0
        for a in list(df_orbital_attributes_of_elements.index):

            ea = df_orbital_attributes_of_elements.loc[a, :] * 1
            features = {}
            for o in math_operators.keys():
                features[a + '.' + o] = {}
            if ea.notnull().sum() == 0:
                # when all atomic_attribute values are nan, their feature_variables should be nan, too.
                for o in math_operators.keys():
                    for cf in chemical_formula_list:
                        features[a + '.' + o][cf] = np.nan
            else:
                for cf in chemical_formula_list:
                    v = ea * elements_existence.loc[cf]
                    w = df_composition.loc[cf]  # weightings
                    wea = v * w

                    # compound_wavg is nan if this atomic_attribute are_empty for all elements.
                    # compound_wavg = [np.nansum(wea) / w.sum(), np.nan][wea.notnull().sum() == 0]
                    compound_wavg = np.nansum(wea) / w.sum() if wea.notnull().sum() else np.nan

                    # compound_sum is nan if this atomic_attribute are_empty for all elements.
                    # compound_sum = [np.nansum(v), np.nan][v.notnull().sum() == 0]
                    compound_sum = np.nansum(v) if v.notnull().sum() else np.nan

                    features[a + '.max'][cf] = np.nanmax(v)
                    features[a + '.min'][cf] = np.nanmin(v)
                    features[a + '.range'][cf] = np.nanmax(v) - np.nanmin(v)
                    features[a + '.std'][cf] = np.nanstd(v)
                    features[a + '.avg'][cf] = np.nanmean(v)
                    features[a + '.wavg'][cf] = compound_wavg
                    features[a + '.sum'][cf] = compound_sum

                    i += 1
                    progressbar(i, df_composition.shape[0] * df_orbital_attributes_of_elements.shape[0]) if _print else 0

            _df_features = pd.DataFrame.from_dict(features, orient='columns', dtype='float64')
            _df_features.to_csv(cache_path + 'feature_variables [' + a + '._].csv', float_format='%8f')

        warnings.resetwarnings

        # reload features
        df_features = pd.DataFrame(dtype='float64')
        file_list = os.listdir(cache_path)
        i = 0
        for f in file_list:
            if os.path.isfile(cache_path + f):
                df_data = pd.read_csv(cache_path + f, index_col=0)
                for feature_name in list(df_data.columns):
                    df_features[feature_name] = df_data[feature_name] * 1
            i += 1
            progressbar(i, len(file_list)) if _print else 0

        # df_features.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'feature_variables.csv', float_format='%8f')

        shutil.rmtree(cache_path)

        print('  Done.') if _print else 0

        return df_features
