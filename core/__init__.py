# coding: utf-8
# Copyright (c) pytmge Development Team.

"""
This package contains core modules and classes
for machine learning to predict materials.

"""

from pytmge.core.plugins import progressbar

from pytmge.core.elemental_data import elemental_data
# from .elemental_data import electron_orbital_attributes_of_elements


_print = [False, True][1]

element_list = elemental_data().symbols
# eoa = electron_orbital_attributes_of_elements()
