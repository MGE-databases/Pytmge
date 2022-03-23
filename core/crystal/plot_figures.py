# -*- coding: utf-8 -*-
"""

"""


import os
import matplotlib.pyplot as plt
from pathlib import Path
from pytmge.core import progressbar, _print


_path = str(Path(__file__).absolute().parent) + '\\figures\\'


def plot_target_vs_features(ds_target, df_features, path=_path):
    '''
    Plot the figures of target_vs_feature.

    Parameters
    ----------
    ds_target : Series
        ds_target.
    df_features : DataFrame
        df_features.
    path : str, optional
        The path saving figure fiels. The default is str(Path(__file__).absolute().parent) + '\\figures\\'.

    Returns
    -------
    None.

    '''

    print('\n  plotting target_vs_feature figures ...') if _print else 0

    for i, f in enumerate(list(df_features)):
        ds_feature = df_features.loc[list(ds_target.index), f]

        plt.rcdefaults()
        plt.style.use('ggplot')
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.figsize'] = (3.3, 3.3)

        plt.figure(facecolor='w', edgecolor='k')

        plt.tick_params(labelsize=12,
                        direction='out',
                        width=0.5,
                        length=2,
                        top=False,
                        right=False)

        plt.xlabel(ds_feature.name.replace(r'_', ' '))  # , fontsize=12
        plt.ylabel(ds_target.name.replace(r'_', ' '))  # or like '$T\\mathregular{_{c}^{max}}$'

        plt.scatter(ds_feature, ds_target,
                    marker='o',
                    color='red',
                    edgecolors='black',
                    s=8,
                    linewidths=0.1,
                    alpha=0.3)

        # plt.show()

        if not(os.path.exists(path)):
            os.makedirs(path)

        plt.savefig(path + 'figure target-feature (' + ds_feature.name + ').png', bbox_inches='tight')

        for fn in plt.get_fignums():
            plt.close(fn)

        progressbar(i + 1, df_features.shape[1]) if _print else 0

    print('  Done.') if _print else 0

    return
