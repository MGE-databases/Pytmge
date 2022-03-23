# -*- coding: utf-8 -*-

"""
An example.

"""


import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from pytmge.core import elemental_data
from pytmge.core.crystal import data_set
from pytmge.core.crystal import feature_design, feature_engineering
from pytmge.core.crystal import plot_target_vs_features


if __name__ == '__main__':

    _path = str(Path(__file__).absolute().parent) + '\\'

    ed = elemental_data()  # load data of elemental attributes

    df_example = pd.read_csv(_path + 'example.csv', index_col=0).iloc[:, :]
    # df_example.sort_index(inplace=True)
    # df_example.sort_values(by=list(df_example)[0], ascending=False, inplace=True)
    df_example.sort_values(by=list(df_example)[0], ascending=True, inplace=True)

    #

    # ------ dataset ------

    dataset = data_set(df_example)

    # df_deduped_dataset = dataset.delete_duplicates()
    # dataset = data_set(df_deduped_dataset)

    chemical_formulas = dataset.chemical_formulas
    composition = chemical_formulas.composition
    df_comp = composition.df
    composition.df.to_csv(_path + 'df_composition.csv')

    # ------ subset ------
    # dict_category = dataset.categorization_by_composition()
    df_subset = dataset.subset()
    # df_subset.to_csv(_path + 'df_subset.csv')

    dataset = data_set(df_subset)
    chemical_formulas = dataset.chemical_formulas
    composition = chemical_formulas.composition

    #

    # ------ features ------

    df_features = feature_design.get_features(composition.df)
    df_features.to_csv(_path + 'df_features.csv')

    df_usable_features = feature_design.delete_unusable_features(df_features)
    df_usable_features.to_csv(_path + 'df_usable_feature.csv')

    # feature selection by Pearson correlation
    df_selected_features = feature_engineering.feature_selection_by_Pearson_correlation(df_usable_features)

    plot_target_vs_features(dataset.target_variable, df_selected_features)

    # Standardization
    df = df_usable_features * 1
    df_standardized_usable_features = pd.DataFrame(
        StandardScaler().fit(df).transform(df),
        index=df.index,
        columns=df.columns
    )
