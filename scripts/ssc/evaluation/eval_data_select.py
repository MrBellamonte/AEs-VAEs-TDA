import os
import shutil

import pandas as pd


def get_latent_rename(exp_root, eval_root, uid, latent_name):
    uid_rood = os.path.join(exp_root, uid)

    existing_file = open(os.path.join(uid_rood, 'train_latent_visualization.pdf'), "r")
    new_file = open(os.path.join(eval_root, latent_name+'.pdf'), "w")

    src_file = os.path.join(uid_rood, 'train_latent_visualization.pdf')
    dest_dir = eval_root

    shutil.copy(src_file, dest_dir)  # copy the file to destination dir

    dst_file = os.path.join(eval_root, 'train_latent_visualization.pdf')
    new_dst_file_name = os.path.join(eval_root, latent_name+'.pdf')

    os.rename(dst_file, new_dst_file_name)  # rename
    os.chdir(dest_dir)


if __name__ == "__main__":

    N = 3

    bss = [64, 128, 256, 512]

    WCTopoAE_symmetric = False
    WCTopoAE_apush = False
    UMAP = False
    UMAP2 = False
    tSNE = False
    TopoAE = False

    UMAP_final = False
    tSNE_final = False
    TopoAE_new = True

    WCAE = False

    if WCTopoAE_symmetric:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCTopoAE_swissroll_symmetric'
        root_save = '/Users/simons/MT_data/eval_all_analysis/WCTopoAE/SwissRoll/symmetric'
    elif WCTopoAE_apush:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCTopoAE_swissroll_apush'
        root_save = '/Users/simons/MT_data/eval_all_analysis/WCTopoAE/SwissRoll/apush'
    elif UMAP:
        exp_dir = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/competitors/swissroll_umap_new2'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/UMAP/swissroll2'
    elif UMAP2:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/umap_swissroll'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/UMAP/final_swissroll'
    elif tSNE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/tsne_swissroll'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/tSNE/final_swissroll'
    elif TopoAE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/TopoAE_swissroll_symmetric'
        root_save = '/Users/simons/MT_data/eval_all_analysis/TopoAE/SwissRoll/symmetric_nonshuffle'
    elif UMAP_final:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/umap_final'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/UMAP/FINAL'
        bss = [0]
    elif tSNE_final:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/tsne_final'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/tSNE/FINAL'
        bss = [0]
    elif WCAE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise'
        df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed_new_new.csv'
        root_save = '/Users/simons/MT_data/eval_all_analysis/WCAE/SwissRoll_nonoise'
    elif TopoAE_new:
        df_path = '/Users/simons/MT_data/eval_all_analysis/TopoAE/SwissRoll/reevaluation/results.csv'
        root_save = '/Users/simons/MT_data/eval_all_analysis/TopoAE/SwissRoll/reevaluation'
    else:
        ValueError

    criterion = 'test_mean_Lipschitz_std_refZ'



    max_metrics = ['test_mean_trustworthiness','test_mean_continuity']


    if WCAE or TopoAE_new:
        df_path = df_path
    else:
        df_path = os.path.join(exp_dir, 'eval_metrics_all_new.csv')
    print('Load data...')
    df = pd.read_csv(df_path)

    if UMAP_final or tSNE_final:
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
        ]
        cols_1 = ['uid', 'metric', 'value', 'seed']
        cols_2 = ['seed']
        cols_3 = ['uid', 'metric', 'value']
        cols_1 = ['uid','batch_size','metric','value','seed']
        cols_2 = ['seed','batch_size']
        cols_3 = ['uid', 'batch_size', 'metric', 'value']
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


    df_criterion_metric = df[cols_1]
    df_criterion_metric = df_criterion_metric[df_criterion_metric.metric == criterion]

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
