import os

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def fancy_name(fancy_mapping: dict):
    return list(fancy_mapping.items())[0][1]


FANCY_recloss = {'training.reconstruction_error': '$\mathcal{L}_{rec}$'}
FANCY_toploss = {'training.loss.topo_error': '$\mathcal{L}_{top}$'}
FANY_pairs = {"training.metrics.matched_pairs_0D" : 'matched pairs'}

FANCY_test_KL_1 = {'test_density_kl_global_1': '$KL_{1}$'}
FANCY_test_KL_01 = {'test_density_kl_global_01': '$KL_{0.1}$'}
FANCY_test_KL_001 = {'test_density_kl_global_001': '$KL_{0.01}$'}
FANCY_test_KL_0001 = {'test_density_kl_global_0001': '$KL_{0.001}$'}

FANCY_test_KL_00001 = {'test_density_kl_global_00001': '$KL_{0.0001}$'}
FANCY_test_KL_000001 = {'test_density_kl_global_000001': '$KL_{0.00001}$'}
FANCY_test_KL_0000001 = {'test_density_kl_global_0000001': '$KL_{0.000001}$'}

FANCY_K5X_min = {"test_K_5X_min": '$K_{5,min}^X$'}
FANCY_K5X_max = {"test_K_5X_max": '$K_{5,max}^X$'}
FANCY_K5X_avg = {"test_K_5X_avg": '$K_{5,avg}^X$'}
FANCY_K5X_min_norm = {"test_K_norm_5X_min": '$K_{5,min,norm}^X$'}
FANCY_K5X_max_norm = {"test_K_norm_5X_max": '$K_{5,max,norm}^X$'}
FANCY_K5X_avg_norm = {"test_K_norm_5X_avg": '$K_{5,avg,norm}^X$'}

FANCY_K5Z_min = {"test_K_5Z_min": '$K_{5,min}^Z$'}
FANCY_K5Z_max = {"test_K_5Z_max": '$K_{5,max}^Z$'}
FANCY_K5Z_avg = {"test_K_5Z_avg": '$K_{5,avg}^Z$'}
FANCY_K5Z_min_norm = {"test_K_norm_5Z_min": '$K_{5,min,norm}^Z$'}
FANCY_K5Z_max_norm = {"test_K_norm_5Z_max": '$K_{5,max,norm}^Z$'}
FANCY_K5Z_avg_norm = {"test_K_norm_5Z_avg": '$K_{5,avg,norm}^Z$'}

FANCY_LRMSE_5X = {"test_llrmse_X5": '$l-RMSE_{5,X}$'}
FANCY_LRMSE_5Z = {"test_llrmse_Z5": '$l-RMSE_{5,Z}$'}
FANCY_LRMSE_5X_norm = {"test_llrmse_X_norm5": '$l-RMSE_{5,X,norm}$'}
FANCY_LRMSE_5Z_norm = {"test_llrmse_Z_norm5": '$l-RMSE_{5,Z,norm}$'}

FANCY_test_mrre = {'test_mrre': 'l-MRRE'}
FANCY_test_rmse = {'test_rmse': 'l-RMSE'}
FANCY_test_cont = {'test_continuity': 'l-Cont'}
FANCY_test_trust = {'test_trustworthiness': 'l-Trust'}

Y_MAPPING = {
    '$\mathcal{L}_{rec}$': 1,
    '$\mathcal{L}_{top}$': 2,
    '$KL_{1}$'           : (1+2),
    '$KL_{0.1}$'         : (2+2),
    '$KL_{0.01}$'        : (3+2),
    '$KL_{0.001}$'       : (4+2),
    'l-MRRE'             : (5+2),
    'l-RMSE'             : (6+2),
    'l-Cont'             : (7+2),
    'l-Trust'            : (8+2)
}

Y_MAPPING2 = {
    '$\mathcal{L}_{rec}$': 1,
    '$\mathcal{L}_{top}$': 2,
    'matched pairs': 3,
    '$KL_{1}$'           : (1+3),
    '$KL_{0.1}$'         : (2+3),
    '$KL_{0.01}$'        : (3+3),
    '$KL_{0.001}$'       : (4+3),
    '$KL_{0.0001}$'  : (5+3),
    '$KL_{0.00001}$' : (6+3),
    '$KL_{0.000001}$': (7+3),
    'l-MRRE'             : (5+-4+10),
    'l-RMSE'             : (6+-4+10),
    'l-Cont'             : (7+-4+10),
    'l-Trust'            : (8+-4+10),
    '$l-RMSE_{5,X}$'     : (6-5+14),
    '$l-RMSE_{5,Z}$'     : (7-5+14),
    '$l-RMSE_{5,X,norm}$': (8-5+14),
    '$l-RMSE_{5,Z,norm}$': (9-5+14),
    '$K_{5,min}^X$' : (18-3+4),
    '$K_{5,max}^X$' : (18-3+5),
    '$K_{5,avg}^X$' : (18-3+6),
    '$K_{5,min,norm}^X$'    : (18-3+7),
    '$K_{5,max,norm}^X$'    : (18-3+8),
    '$K_{5,avg,norm}^X$'    : (18-3+9),
    '$K_{5,min}^Z$' : (18-3+10),
    '$K_{5,max}^Z$' : (18-3+11),
    '$K_{5,avg}^Z$' : (18-3+12),
    '$K_{5,min,norm}^Z$'    : (18-3+13),
    '$K_{5,max,norm}^Z$'    : (18-3+14),
    '$K_{5,avg,norm}^Z$'    : (18-3+15),

}

mapping = [FANCY_recloss, FANCY_toploss, FANCY_test_KL_1, FANCY_test_KL_01, FANCY_test_KL_001,
           FANCY_test_KL_0001, FANCY_test_mrre, FANCY_test_rmse, FANCY_test_cont, FANCY_test_trust]
mapping2 = mapping+[FANY_pairs, FANCY_test_KL_00001, FANCY_test_KL_000001, FANCY_test_KL_0000001, FANCY_K5X_min,
                    FANCY_K5X_max, FANCY_K5X_avg, FANCY_K5X_min_norm, FANCY_K5X_max_norm,
                    FANCY_K5X_avg_norm, FANCY_K5Z_min, FANCY_K5Z_max, FANCY_K5Z_avg,
                    FANCY_K5Z_min_norm, FANCY_K5Z_max_norm, FANCY_K5Z_avg_norm, FANCY_LRMSE_5X,
                    FANCY_LRMSE_5Z,
                    FANCY_LRMSE_5X_norm,
                    FANCY_LRMSE_5Z_norm]
if __name__ == "__main__":
    df_path = '/Users/simons/polybox/Studium/20FS/MT/sync/euler_sync/schsimo/MT/output/TopoAE/SwissRoll/multiseed/eval_metrics_all.csv'
    df_path2 = '/Users/simons/polybox/Studium/20FS/MT/sync/euler_sync/schsimo/MT/output/TopoAE/SwissRoll/eval_verification/eval_metrics_all.csv'
    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/multiseed'


    uid_exceptional = ['SwissRoll-n_samples2560-Autoencoder_MLP_topoae-32-32-lr1_1000-bs256-nep1000-rlw1-tlw4096-seed102-7f02781d']

    uid_exceptional2 = [
        'SwissRoll-n_samples2560-Autoencoder_MLP_topoae-32-32-lr1_1000-bs256-nep1000-rlw1-tlw4096-seed102-4ce6955a',
        'SwissRoll-n_samples2560-Autoencoder_MLP_topoae-32-32-lr1_1000-bs256-nep1000-rlw1-tlw8192-seed102-3f14a4fb'
    ]



    new = False

    if new:
        mapping = mapping2
        Y_MAPPING = Y_MAPPING2
        df = pd.read_csv(df_path2)
        name2 = 'metric_comparison2'
    else:
        uid_exceptional2 = uid_exceptional
        df = pd.read_csv(df_path)
        name2 = 'metric_comparison1'

    normalization = False
    standardize = True
    selection = []
    for element in mapping:
        selection.append(fancy_name(element))


    selection = list(Y_MAPPING.keys())
    name_replace = dict()
    for dictionary in mapping:
        name_replace.update(dictionary)
    df = df.replace(name_replace)
    df = df[df['metric'].isin(selection)]

    if standardize:
        y_label = 'standardized metric value'
        for metric in selection:
            std = df.loc[df['metric'] == metric, 'value'].std()
            mean = df.loc[df['metric'] == metric, 'value'].median()
            df.loc[df['metric'] == metric, 'value'] = (df.loc[df[
                                                                  'metric'] == metric, 'value']-mean)/std

    if normalization:
        y_label = 'normalized metric value'
        for metric in selection:
            min = df.loc[df['metric'] == metric, 'value'].min()
            max = df.loc[df['metric'] == metric, 'value'].max()
            df.loc[df['metric'] == metric, 'value'] = (df.loc[
                                                           df['metric'] == metric, 'value']-min)/(
                                                              max-min)

    # SET X-VALUE CORRESPONDING TO METRIC
    df['x'] = df.loc[:, 'metric'].replace(Y_MAPPING)

    # SET MARKER
    df['cat'] = 0
    df.loc[df['uid'].isin(uid_exceptional2), 'cat'] = 1
    fig, ax = plt.subplots()
    sns.scatterplot(data=df[df['cat'] != 1], x="x", y="value", color='black', ax = ax, label = 'rest')
    sns.scatterplot(data=df[df['cat'] == 1], x="x", y="value", color='red',ax = ax, label = 'good example')
    plt.xticks(np.arange(1, (len(selection)+1)), selection,rotation='vertical')
    plt.ylabel(y_label)
    plt.xlabel('')
    plt.show()
    if normalization:
        name = 'normalized'

    if standardize:
        name = 'standardized'

    fig.savefig(os.path.join(path_to_save, '{}_{}.pdf'.format(name,name2)), dpi=200)
