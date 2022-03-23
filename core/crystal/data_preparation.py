# coding: utf-8
# Copyright (c) pytmge Development Team.

'''
Classes for data preparation.
    Dataset
    chemical_formulas
    composition

'''

import re
import numpy as np
import pandas as pd

from pytmge.core import element_list, progressbar, _print


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


class data_set:

    def __init__(self, df_dataset):
        '''
        df_dataset : DataFrame
            Dataset of chemical formula and target variable.
            (chemical formulas as index,
            target variable and labels as columns,
            target variable at the first column)

        '''

        self.data = df_dataset
        self.chemical_formulas = chemical_formulas(df_dataset)
        self.target_variable = df_dataset.iloc[:, 0]

    def delete_duplicates(self):
        '''
        Delete duplicate entries in dataset.

        If two or more entries have the same chemical formula
        but different property values, keep the entry having greater value (of the first column).

        Returns
        -------
        df_deduped_dataset : DataFrame
            deduped subset.

        '''

        print('\n  deleting duplicate entries in dataset ...') if _print else 0

        # sort the dataset by the values of columns (first column at the end).
        for col_name in list(self.data)[::-1]:
            self.data.sort_values(by=col_name, inplace=True)

        deduped_dataset = {}
        for i, cf in enumerate(list(self.data.index)):
            deduped_dataset[cf] = self.data.loc[cf]
            progressbar(i + 1, self.data.shape[0]) if _print else 0

        print('  original:', i + 1, '| deduped:', len(deduped_dataset)) if _print else 0

        df_deduped_dataset = pd.DataFrame.from_dict(deduped_dataset, orient='index')
        # df_deduped_dataset.sort_index(ascending=True, inplace=True)
        df_deduped_dataset.sort_values(
            by=list(df_deduped_dataset)[0],
            ascending=False,
            inplace=True
        )

        print('  Done.') if _print else 0

        return df_deduped_dataset

    def categorization_by_composition(self):
        '''
        Categorizing the chemical formulas, according to (n-e-c).
            n : 'number_of_elements',
            e : 'element',
            c : 'elemental_contents'

        Returns
        -------
        dict_category : dict
            Dict of category.

        '''

        print('\n  categorizing chemical formulas ...') if _print else 0

        df_composition = self.chemical_formulas.composition.data['DataFrame']
        cfs = list(df_composition.index)
        _elements = list(df_composition.columns)

        # elemental contents
        contents = df_composition.fillna(0).applymap(lambda x: np.int(x + 0.5))

        # number of elements in each chemical formula, ignore the element(s) that content < 0.5
        number_of_elements = (df_composition.applymap(lambda x: x >= 0.5) * 1).sum(axis=1)

        # print('  labeling ...') if _print else 0

        category_labels = {}
        for cf in cfs:
            category_labels[cf] = {}
            i = 0
            for e in _elements:
                if contents.loc[cf, e] >= 0.5:
                    # assign a category lable 'n-e-c' to each chemical formula
                    n = str(round(number_of_elements[cf]))
                    c = str(int(contents.loc[cf, e] + 0.5))
                    category_labels[cf][i] = n + '-' + e + '-' + c
                    i += 1

        # print('  collecting ...') if _print else 0

        dict_category = {}
        for cf in cfs:
            for label in category_labels[cf].values():
                # dict_category[label] = {}
                dict_category[label] = []
        for cf in cfs:
            for label in category_labels[cf].values():
                # dict_category[label][cf] = self.data.loc[cf, :].to_dict()
                dict_category[label] += (cf, )

        print('  Done.') if _print else 0

        return dict_category

    def subset(self):
        '''
        Extracting subset.
        For each category, pick one entry having the highest value of material property.

        Returns
        -------
        df_subset : DataFrame
            df_subset.

        '''

        dict_category = self.categorization_by_composition()

        print('\n  extracting subset ...') if _print else 0

        ds_dataset = self.data.iloc[:, 0]

        highest_entries = {}
        for category_label, cfs in dict_category.items():
            highest_value = ds_dataset[cfs].nlargest(1).values[0]
            highest_entries[category_label] = ds_dataset[cfs][ds_dataset >= highest_value * 1.0].to_dict()
            # top_3 = ds_dataset[cfs].nlargest(3).index
            # highest_entries[category_label] = ds_dataset[cfs][top_3].to_dict()

        list_highest = []
        for k, v in highest_entries.items():
            list_highest += list(v)  # sometimes there are multiple highest ones
            # list_highest += (list(v)[0], )  # only take one

        df_subset = self.data.loc[list(set(list_highest)), :]

        df_subset.sort_values(by=list(df_subset)[0], ascending=False, inplace=True)

        print('  Done.') if _print else 0

        return df_subset


class chemical_formulas:

    def __init__(self, dataset: object):
        self.data = list(dataset.index)
        self.in_proper_format = self.check_format()
        self.composition = composition(self.in_proper_format)

    def check_format(self):
        '''
        Checking the format of the chemical formulas.

        The format is supposed to be like 'H2O1' or 'C60',
        whereas 'H2O' or 'C' or 'La2Cu1O4-x' is NOT ok.

        Do NOT use brakets.

        Returns
        -------
        chemical_formulas_in_proper_format : list
            Chemical formulas in proper format.

        '''

        print('\n  checking format of chemical formulas ...') if _print else 0

        chemical_formulas_in_proper_format = []
        is_proper_format = {}

        for i, cf in enumerate(self.data):

            is_proper_format[cf] = True

            if pd.isnull(cf):
                is_proper_format[cf] = False
                print('  chemical formula No.', i + 1, 'is null ...')
            else:
                try:
                    elements_in_cf = re.split(r'[(0-9]*[\.]?[0-9]+', cf)
                    if elements_in_cf[-1] == '':
                        elements_in_cf = elements_in_cf[:-1]
                    else:
                        is_proper_format[cf] = False
                    if elements_in_cf == '':
                        is_proper_format[cf] = False
                    for e in elements_in_cf:
                        if e not in element_list:
                            is_proper_format[cf] = False
                            break
                except Exception:
                    is_proper_format[cf] = False

            if is_proper_format[cf]:
                chemical_formulas_in_proper_format += (cf, )
            else:
                print('\n  chemical formula seems not right :', cf) if _print else 0

        print('  ' + str(len(self.data) - len(chemical_formulas_in_proper_format)),
              'chemical formulas seem not right.') if _print else 0

        print('  Done.') if _print else 0

        return chemical_formulas_in_proper_format


class composition:

    def __init__(self, chemical_formulas: list):
        self._chemical_formulas = chemical_formulas
        self.data = self.composition()
        self.df = self.data['DataFrame']
        self.dict = self.data['dict']
        self.alloys = self.data['alloys']  # the contents of which were divided by 100.

    def composition(self):
        '''
        Extracting composition from chemical formulas.

        If the cf is an alloy, the sum of contents is close to 100, then normalize to 1.

        Parameters
        ----------
        chemical_formulas : list
            List of chemical formulas.

        Returns
        -------
        dict_composition : dict
            Chemical formulas as keys, {element: content} as values.

        df_composition : DataFrame
            Chemical formulas as index, elements as columns.

        alloys : list
            A list of alloys, the contents of which were divided by 100.
        '''

        print('\n  extracting composition of chemical formulas ...') if _print else 0

        dict_composition = {}
        alloys = []

        for i, cf in enumerate(self._chemical_formulas):

            elements_in_cf = re.split(r'[(0-9]*[\.]?[0-9]+', cf)
            contents_in_cf = list(map(float, re.findall(r'[0-9]*[\.]?[0-9]+', cf)))

            # if the cf is an alloy, the sum of contents is close to 100, then normalize to 1.
            # if 99.5 <= np.nansum(contents_in_cf) <= 100.5:
            if np.nansum(contents_in_cf) == 100:

                contents_in_cf = list(pd.Series(contents_in_cf) / 100)
                alloys += (cf, )
                # print('contents divided by 100 :', cf)

            dict_composition[cf] = {}
            for e, c in zip(elements_in_cf, contents_in_cf):
                dict_composition[cf][e] = dict_composition[cf].get(e, 0.0) + c
                # Note: sometimes some elements appear multiple times in a cf.

            progressbar(i + 1, len(self._chemical_formulas)) if _print else 0

        print('  getting DataFrame from dict ...') if _print else 0
        df_0 = pd.DataFrame(index=list(dict_composition), columns=element_list).fillna(0)
        df_composition = pd.DataFrame.from_dict(dict_composition, orient='index')
        df_composition = (df_0 + df_composition).replace(0, np.nan)
        df_composition = df_composition.loc[self._chemical_formulas, element_list] * 1
        dict_composition = df_composition.fillna(0).to_dict(orient='index')

        # print('saving...')
        # df_composition.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'df_composition.csv')
        # print('df_composition.csv')
        # with open(_path + 'dict_composition.json', 'w') as _f:
        #     json.dump(dict_composition, _f)
        # print('dict_composition.json')

        print('  Done.') if _print else 0

        return {'dict': dict_composition, 'DataFrame': df_composition, 'alloys': alloys}
