import pandas as pd


def normalize_metric(df, metric_col, metric_norm_name):

    df_norm = df[df['metric'] == metric_col].loc[:,:]

    max_val = df_norm.loc[:,'value'].max()
    min_val = df_norm.loc[:,'value'].min()

    df_norm[metric_norm_name] = (df_norm.loc[:,'value'] - min_val)/(max_val-min_val)

    return df_norm[['uid',metric_norm_name]]


def average_metrics(df, cols, new_col_name):
    df[new_col_name] = df.loc[:,cols].sum(axis=1)/len(cols)
    return df



if __name__ == "__main__":
    df_path = '/Users/simons/polybox/Studium/20FS/MT/sync/euler_sync/schsimo/MT/output/TopoAE/SwissRoll/multiseed/eval_metrics_all.csv'

    path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/TopoAE/SwissRoll/multiseed/multimetric_eval/eval_multimetrics_all.csv'

    df = pd.read_csv(df_path)

    df_norm_KL001 = normalize_metric(df, 'test_density_kl_global_001', 'kl_001_norm')
    df_norm_MRRE = normalize_metric(df, 'test_mrre', 'mrre_norm')
    df_norm_Kmax = normalize_metric(df, 'test_K_max', 'Kmax_norm')


    df_joined = df_norm_KL001.set_index('uid').join(df_norm_MRRE.set_index('uid')).join(df_norm_Kmax.set_index('uid'))

    df_joined = average_metrics(df_joined, ['kl_001_norm','mrre_norm'],'KL_MRRE')
    df_joined = average_metrics(df_joined, ['kl_001_norm','Kmax_norm'], 'KL_Kmax')
    df_joined = average_metrics(df_joined, ['mrre_norm','Kmax_norm'], 'MRRE_Kmax')
    df_final = average_metrics(df_joined, ['kl_001_norm','mrre_norm','Kmax_norm'], 'all')

    print(df_final.head())

    df_final.to_csv(path_to_save)







