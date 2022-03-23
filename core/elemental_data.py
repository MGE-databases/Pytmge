# coding: utf-8
# Copyright (c) pytmge Development Team.

"""
Providing data of fundamental attributes of elements.

"""


import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


_data_path = str(Path(__file__).absolute().parent) + '\\'


class elemental_data():

    def __init__(self):

        self.orbital_attributes_of_shells = self._orbital_attributes_of_shells()
        self.orbital_attributes_of_elements = self._orbital_attributes_of_elements()
        self.atomic_attributes_of_elements = self._atomic_attributes_of_elements()
        self.occupancy_of_electron_shells = self._occupancy_of_electron_shells()
        self.energy_level_of_electron_shells = self._energy_level_of_electron_shells()
        self.symbols = list(self.occupancy_of_electron_shells)

    def _orbital_attributes_of_shells(self):
        with open(_data_path + "orbital_attributes_of_shells.json", "rt") as _f:
            _data = json.load(_f)
        return _data

    def _orbital_attributes_of_elements(self):
        with open(_data_path + "orbital_attributes_of_elements.json", "rt") as _f:
            _data = json.load(_f)
        # df_orbital_attributes_of_elements = pd.DataFrame.from_dict(_data, orient='index')
        # return df_orbital_attributes_of_elements
        return _data

    def _atomic_attributes_of_elements(self):
        with open(_data_path + "atomic_attributes_of_elements.json", "rt") as _f:
            _data = json.load(_f)
        return _data

    def _occupancy_of_electron_shells(self):
        # Herman, F., Sherwood Skillman, S. and Arents, J. Atomic Structure Calculations. Vol. 111 (Prentice-Hall, 1964).
        with open(_data_path + "occupancy_of_electron_shells.json", "rt") as _f:
            _data = json.load(_f)
        return _data

    def _energy_level_of_electron_shells(self):
        # Herman, F., Sherwood Skillman, S. and Arents, J. Atomic Structure Calculations. Vol. 111 (Prentice-Hall, 1964).
        with open(_data_path + "energy_level_of_electron_shells.json", "rt") as _f:
            _data = json.load(_f)
        return _data


class electron_orbital_attributes_of_elements():
    '''
    Extracting electron orbital attributes of each element.
    A elemental_attributes = [attribute].[_shell_selection].[math operator]

    '''

    def __init__(self):

        self._data_source = '[Herman, F., Sherwood Skillman, S. and Arents, J. Atomic Structure Calculations. Vol. 111 (Prentice-Hall, 1964).]'
        self._shell_occupancy = pd.DataFrame.from_dict(elemental_data().occupancy_of_electron_shells, orient='index')
        self._shell_energy = pd.DataFrame.from_dict(elemental_data().energy_level_of_electron_shells, orient='index')
        self._elements = list(self._shell_occupancy.index)
        self._shells = list(self._shell_occupancy.columns)

        self._valence = self._get_valence()  # energy_threshold = -36 eV
        self._valence_energy = self._shell_energy * self._valence  # energy level
        self._valence_number_of_allowed = self._get_valence_number_of_allowed()
        self._valence_number_of_filled = self._shell_occupancy * self._valence  # filled number
        self._valence_number_of_unfilled = self._valence_number_of_allowed - self._valence_number_of_filled  # unfilled number of _valence shells
        self._valence_filling_rate = self._valence_number_of_filled / self._valence_number_of_allowed
        self._valence_filling_saturation = self._get_valence_filling_saturation()  # is fully filled ?
        self._valence_filling_parity = self._valence_number_of_filled % 2  # filling parity (odd or even)
        self._valence_main_quantum_number = self._get_valence_main_quantum_number()
        self._valence_angular_quantum_number = self._get_valence_angular_quantum_number()

        self._attributes = self._get_attributes()
        self._shell_selection = self._get_shell_selection()

        self.shell_attributes = self._get_shell_attributes()
        self.elemental_attributes = self._get_elemental_attributes()

    def _get_valence(self):
        energy_threshold = -36  # the shells in [0, -36] (eV) are considered as valence shells.
        return (self._shell_energy.applymap(lambda x: x >= energy_threshold) * 1).replace(0, np.nan)

    def _get_valence_number_of_allowed(self):
        list_a = [2, 2, 6, 2, 6, 10, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 2]  # 18 shells in total
        df_a = pd.DataFrame(pd.Series(list_a, index=self._shells)).T
        df_1 = pd.DataFrame(index=self._elements, columns=['shells']).fillna(1)
        allowed = pd.DataFrame(np.dot(df_1, df_a), index=self._elements, columns=self._shells)
        return self._valence * allowed

    def _get_valence_filling_saturation(self):
        return self._valence_filling_rate.fillna(0).applymap(lambda x: int(x)) * self._valence

    def _get_valence_main_quantum_number(self):
        _main_quantum_number = self._valence.apply(
            lambda x:
                pd.Series(
                    [int(self._shells[i][0])
                     if x[i] == 1
                     else np.nan
                     for i in list(range(18))],
                    index=self._shells
                ),
            axis=1
        )
        return _main_quantum_number

    def _get_valence_angular_quantum_number(self):
        _angular_quantum_number = self._valence.apply(
            lambda x:
                pd.Series(
                    [['s', 'p', 'd', 'f'].index(self._shells[i][1])
                     if x[i] == 1
                     else np.nan
                     for i in list(range(18))],
                    index=self._shells
                ),
            axis=1
        )
        return _angular_quantum_number

    def _get_attributes(self):
        '''
        Getting attributes of all _valence orbitals.

        Returns
        -------
        attributes : dict
            Each of its values is a Pandas DataFrame (index=elemnts, columns=shells),

        '''

        attributes = {}
        attributes['E'] = self._valence_energy
        attributes['Nf'] = self._valence_number_of_filled
        attributes['Nu'] = self._valence_number_of_unfilled

        attributes['Fr'] = self._valence_filling_rate
        attributes['Fs'] = self._valence_filling_saturation
        attributes['Fp'] = self._valence_filling_parity

        attributes['n'] = self._valence_main_quantum_number
        attributes['l'] = self._valence_angular_quantum_number

        return attributes

    def _get_shell_selection(self):
        '''
        Assigning which shells are considered.

        Returns
        -------
        _shell_selection : dict
            Each of its values is a Pandas DataFrame (index=elemnts, columns=shells),
            in which the selected shells are 1, else are 0.

        '''

        _shell_selection = {}

        aqn = self._valence_angular_quantum_number
        nu = self._valence_number_of_unfilled

        # all _valence
        _shell_selection['all'] = self._valence * 1

        # single shell
        _shell_selection['s'] = aqn.applymap(lambda x: 1 if x == 0 else np.nan)
        _shell_selection['p'] = aqn.applymap(lambda x: 1 if x == 1 else np.nan)
        _shell_selection['d'] = aqn.applymap(lambda x: 1 if x == 2 else np.nan)
        _shell_selection['f'] = aqn.applymap(lambda x: 1 if x == 3 else np.nan)

        # sat: fully occupied
        # unsat: not fully occupied
        _shell_selection['sat'] = (nu.applymap(lambda x: x == 0) * 1).replace(0, np.nan)
        _shell_selection['unsat'] = (nu.applymap(lambda x: x > 0) * 1).replace(0, np.nan)

        # outer: outer shells in real space (s and p shells)
        # inner: inner shells in real space (d and f shells)
        _shell_selection['outer'] = aqn.applymap(lambda x: 1 if x <= 1 else np.nan)
        _shell_selection['inner'] = aqn.applymap(lambda x: 1 if x >= 2 else np.nan)

        return _shell_selection

    def _get_shell_attributes(self):
        '''
        Getting shell attributes.

        A shell_attribute = [attribute].[_shell_selection]

        Parameters
        ----------
        attributes : dict
            Attributes of all _valence orbitals.
        _shell_selection : dict
            Which shells are considered.

        Returns
        -------
        shell_attributes : dict
            Attributes of selected shells.

        '''

        shell_attributes = {}
        for oa, df_orbital_attributes in self._attributes.items():
            for ss, df__shell_selection in self._shell_selection.items():
                shell_attributes[oa + '.' + ss] = df_orbital_attributes * df__shell_selection

        dict_shell_attributesa = {}
        for _name, _attr in shell_attributes.items():
            dict_shell_attributesa[_name] = _attr.to_dict(orient='index')
        with open(_data_path + 'orbital_attributes_of_shells.json', 'w') as _f:
            json.dump(dict_shell_attributesa, _f)

        return shell_attributes

    def _get_elemental_attributes(self):
        '''
        Getting elemental attributes.

        A elemental_attribute = [attribute].[_shell_selection].[math operator]

        Returns
        -------
        elemental_attributes : dict
            Elemental attributes.

        '''

        warnings.filterwarnings('ignore')

        elemental_attributes = {}
        for name_of_shell_attribute, shell_attribute in self.shell_attributes.items():

            atomic_avg = np.nanmean(shell_attribute, axis=1)  # atomic_avg = nan when all_shells_are_empty.
            atomic_std = np.nanstd(shell_attribute, axis=1)
            atomic_max = np.nanmax(shell_attribute, axis=1)
            atomic_min = np.nanmin(shell_attribute, axis=1)
            atomic_range = atomic_max - atomic_min

            is_nan = atomic_avg - atomic_avg  # if all_shells_are_empty is_nan = nan, else is_nan = 0.
            atomic_sum = np.nansum(shell_attribute, axis=1) + is_nan  # when the whole row is empty, the atomic_sum is nan.
            weights = self._valence_number_of_filled * self._shell_selection[name_of_shell_attribute.split('.')[1]]
            atomic_wavg = np.nansum(shell_attribute * weights, axis=1) / weights.sum(axis=1) + is_nan  # when the whole row is empty, the atomic_wavg is nan.

            ds_atomic_avg = pd.Series(np.round(atomic_avg, 6), index=self._elements, dtype='float64')
            ds_atomic_std = pd.Series(np.round(atomic_std, 6), index=self._elements, dtype='float64')
            ds_atomic_max = pd.Series(np.round(atomic_max, 6), index=self._elements, dtype='float64')
            ds_atomic_min = pd.Series(np.round(atomic_min, 6), index=self._elements, dtype='float64')
            ds_atomic_range = pd.Series(np.round(atomic_range, 6), index=self._elements, dtype='float64')
            ds_atomic_sum = pd.Series(np.round(atomic_sum, 6), index=self._elements, dtype='float64')
            ds_atomic_wavg = pd.Series(np.round(atomic_wavg, 6), index=self._elements, dtype='float64')

            elemental_attributes[name_of_shell_attribute + '.avg'] = ds_atomic_avg.to_dict()
            elemental_attributes[name_of_shell_attribute + '.std'] = ds_atomic_std.to_dict()
            elemental_attributes[name_of_shell_attribute + '.max'] = ds_atomic_max.to_dict()
            elemental_attributes[name_of_shell_attribute + '.min'] = ds_atomic_min.to_dict()
            elemental_attributes[name_of_shell_attribute + '.range'] = ds_atomic_range.to_dict()
            elemental_attributes[name_of_shell_attribute + '.sum'] = ds_atomic_sum.to_dict()
            elemental_attributes[name_of_shell_attribute + '.wavg'] = ds_atomic_wavg.to_dict()

        with open(_data_path + 'elemental_attributes.json', 'w') as _f:
            json.dump(elemental_attributes, _f)

        warnings.resetwarnings

        return elemental_attributes
