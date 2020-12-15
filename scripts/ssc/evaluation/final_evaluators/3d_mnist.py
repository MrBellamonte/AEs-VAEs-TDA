import os
import shutil

import pandas as pd
import numpy as np


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

    TopoAE_3D = True



    all_methods = {
        'TopoAE' : '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_topoae_1_deepae3',
        'AE': '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_ae_1_deepae3',
        'UMAP': '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/umap_mnist2',
        'tSNE': '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_tsne'
    }



    methods = all_methods
    df_END = pd.DataFrame()

    path_end = '/Users/simons/MT_data/eval_data/MNIST_FINAL/eval/mnist3d_selected.csv'
    criterions = ['test_mean_Lipschitz_std_refZ','test_mean_local_rmse_refZ','training.reconstruction_error','test_mean_continuity','test_density_kl_global_01']
    max_metrics = ['test_mean_trustworthiness','test_mean_continuity','train_mean_trustworthiness','train_mean_continuity']

    criterions_mapping = {
        'test_mean_Lipschitz_std_refZ' : 'train_mean_Lipschitz_std_refZ',
        'test_mean_local_rmse_refZ' : 'train_mean_local_rmse_refZ',
        'training.reconstruction_error' : 'training.reconstruction_error',
        'test_mean_continuity' : 'train_mean_continuity',
        'test_density_kl_global_01' : 'train_density_kl_global_01'
    }


    for criterion_ in criterions:
        for method in list(methods.keys()):
            criterion = criterion_





            df_path = os.path.join(methods[method], 'eval_metrics_all.csv')
            print('Load data...')
            df = pd.read_csv(df_path)

            if method in ['UMAP', 'tSNE']:
                if criterion == 'training.reconstruction_error':
                    continue_ = False
                else:
                    continue_ = True
                criterion = criterions_mapping[criterion]
                metrics = {
                    'train_mean_Lipschitz_std_refZ' : 'Lipschitz_std_refZ',
                    'train_mean_Lipschitz_std_refX': 'Lipschitz_std_refX',
                    'train_mean_local_rmse_refX': 'local_rmse_refX',
                    'train_mean_local_rmse_refZ': 'local_rmse_refZ',
                    'train_mean_trustworthiness': 'trustworthiness',
                    'train_mean_continuity': 'continuity',
                    'train_density_kl_global_10': 'kl_global_10',
                    'train_density_kl_global_1': 'kl_global_1',
                    'train_density_kl_global_01': 'kl_global_01',
                    'train_density_kl_global_001': 'kl_global_001',
                    'train_density_kl_global_0001': 'kl_global_0001',
                    'train_density_kl_global_00001': 'kl_global_00001'}
            else:
                metrics = {
                    'test_mean_Lipschitz_std_refZ' : 'Lipschitz_std_refZ',
                    'test_mean_Lipschitz_std_refX' :'Lipschitz_std_refX',
                    'test_mean_local_rmse_refX' : 'local_rmse_refX',
                    'test_mean_local_rmse_refZ' : 'local_rmse_refZ',
                    'test_mean_trustworthiness' : 'trustworthiness',
                    'test_mean_continuity' : 'continuity',
                    'test_density_kl_global_10' : 'kl_global_10',
                    'test_density_kl_global_1' : 'kl_global_1',
                    'test_density_kl_global_01' : 'kl_global_01',
                    'test_density_kl_global_001' : 'kl_global_001',
                    'test_density_kl_global_0001' : 'kl_global_0001',
                    'test_density_kl_global_00001' : 'kl_global_00001',
                    'training.reconstruction_error': 'reconstruction_error'}
                continue_ = True

            if continue_:
                cols_criterion = ['uid', 'metric', 'value']
                cols_1 = cols_criterion
                df_criterion_metric = df[cols_criterion]
                df_criterion_metric = df_criterion_metric[df_criterion_metric.metric == criterion]

                if criterion in max_metrics:
                    df_opt = df_criterion_metric.sort_values('value', ascending=False).head(1)
                else:
                    df_opt = df_criterion_metric.sort_values('value', ascending=True).head(1)

                uid_opt = list(df_opt.uid.values)
                df_final = pd.DataFrame(index=np.arange(1), columns=['method'])
                df_final['method'] = method
                df_final['id'] = uid_opt[0]
                df_final['criterion'] = criterion


                df_selected = df[df.uid.isin(uid_opt)]
                for metric in list(metrics.keys()):
                    df_final[metrics[metric]] = df_selected[df_selected['metric']==metric].value.values[0]
                if method in ['UMAP', 'tSNE']:
                    df_final['reconstruction_error'] = 0

                df_END = df_END.append(df_final)




    df_END.to_csv(path_end)


