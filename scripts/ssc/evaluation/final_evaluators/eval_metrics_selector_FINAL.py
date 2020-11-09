import os

import pandas as pd


if __name__ == "__main__":

    N = 3
    bss = [64, 128, 256, 512]

    SWISSROLL_TOPOAE = False
    SWISSROLL_WCAE = False
    SWISSROLL_WCAE2 = False
    UMAP_final = False
    tSNE_final = False
    vanillaAE = True



    if SWISSROLL_TOPOAE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/TopoAE_SwissRoll_final'
        root_save = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/TopoAE_SwissRoll_final'
        df_path = os.path.join(exp_dir,'eval_metrics_all.csv')
        bss = [64, 128, 256, 512]
    elif SWISSROLL_WCAE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise_FINAL'
        root_save = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise_FINAL'
        df_path = os.path.join(exp_dir,'eval_metrics_all.csv')
        bss = [64, 128, 256, 512]
    elif UMAP_final:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/umap_final_final'
        root_save = '/Users/simons/MT_data/sync/euler_sync_scratch/umap_final_final'
        df_path = os.path.join(exp_dir,'eval_metrics_all.csv')
        bss = [0]
    elif tSNE_final:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/tsne_final_final'
        root_save = '/Users/simons/MT_data/sync/euler_sync_scratch/tsne_final_final'
        df_path = os.path.join(exp_dir,'eval_metrics_all.csv')
        bss = [0]
    elif vanillaAE:
        exp_dir ='/Users/simons/MT_data/sync/euler_sync_scratch/vanillaAE_SwissRoll_FINAL'
        root_save = exp_dir
        df_path = os.path.join(exp_dir, 'eval_metrics_all.csv')
        bss = [64, 128, 256, 512]
    else:
        ValueError




    print('Load data...')
    df = pd.read_csv(df_path)


    if UMAP_final or tSNE_final:
        metrics = [
            'rmse_manifold_Z',
            'train_mean_Lipschitz_std_refZ',
            'train_mean_Lipschitz_std_refX',
            'train_mean_local_rmse_refX',
            'train_mean_local_rmse_refZ',
            'train_mean_trustworthiness',
            'train_mean_continuity',
            'train_density_kl_global_10',
            'train_density_kl_global_1',
            'train_density_kl_global_01',
            'train_density_kl_global_001',
            'train_density_kl_global_0001',
            'train_density_kl_global_00001',
        ]
        df['batch_size'] = 0
        cols_1 = ['uid','batch_size','metric','value','seed']
        cols_2 = ['seed','batch_size']
        cols_3 = ['uid','batch_size', 'metric', 'value']

        criterion = 'train_mean_Lipschitz_std_refZ'
        max_metrics = ['train_mean_trustworthiness', 'train_mean_continuity']
    else:
        metrics = [
            'rmse_manifold_Z',
            'test_mean_Lipschitz_std_refZ',
            'test_mean_Lipschitz_std_refX',
            'test_mean_local_rmse_refX',
            'test_mean_local_rmse_refZ',
            'test_mean_trustworthiness',
            'test_mean_continuity',
            'test_density_kl_global_10',
            'test_density_kl_global_1',
            'test_density_kl_global_01',
            'test_density_kl_global_001',
            'test_density_kl_global_0001',
            'test_density_kl_global_00001',
            'training.reconstruction_error'
        ]

        cols_1 = ['uid','batch_size','metric','value','seed']
        cols_2 = ['seed','batch_size']
        cols_3 = ['uid', 'batch_size', 'metric', 'value']

        criterion = 'test_mean_Lipschitz_std_refZ'
        max_metrics = ['test_mean_trustworthiness', 'test_mean_continuity']



    df_criterion_metric = df[df.metric == criterion]
    df_criterion_metric['seed'] = 0


    if tSNE_final or UMAP_final:
        for uuid in list(set(list(df_criterion_metric.uid))):
            df_criterion_metric.loc[(df_criterion_metric.uid == uuid), ['seed']] = int(
            uuid.split('-')[6][4:])



    else:
        for uuid in list(set(list(df_criterion_metric.uid))):
            df_criterion_metric.loc[(df_criterion_metric.uid == uuid), ['seed']] = int(uuid.split('-')[10][4:])


    df_criterion_metric = df_criterion_metric[cols_1]
    df_selected = df_criterion_metric.sort_values('value', ascending=True).groupby(cols_2).head(1)


    df_opt = df_criterion_metric.sort_values('value', ascending=True).groupby(
            ['batch_size']).head(1)

    uid_opt = list(df_opt.uid.values)

    df = df[cols_3]
    uid_selected = list(df_selected.uid.values)

    df_data = df[df.uid.isin(uid_selected)]


    df_final = pd.DataFrame()
    df_final['batch_size'] = bss
    df_final['method'] = 'WCAE'
    i = 0
    for metric in metrics:

        df_temp = df_data[df_data['metric']==metric]


        df_temp = df_temp[['uid','batch_size','value']]

        opt_value = df_temp[df_temp.uid.isin(uid_opt)]
        opt_value = opt_value[['batch_size', 'value']].set_index('batch_size').rename(columns={"value": "{}_opt".format(metric)})
        df_temp = df_temp[['batch_size', 'value']]
        df_temp['value'] = df_temp['value'].astype(float)


        mean_value = df_temp.groupby('batch_size').mean().rename(columns={"value": "{}_mean".format(metric)})
        std_value = df_temp.groupby('batch_size').std().rename(columns={"value": "{}_std".format(metric)})

        if i == 0:
            df_final = opt_value.join(mean_value.join(std_value))
        else:
            df_final = df_final.join(opt_value.join(mean_value.join(std_value)))

        i += 1



    df_data.to_csv(os.path.join(root_save,'selected_data.csv'))
    df_final.to_csv(os.path.join(root_save, 'final_data.csv'))
