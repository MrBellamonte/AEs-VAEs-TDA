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

    WCTopoAE_symmetric = False
    WCTopoAE_apush = False
    UMAP = False
    UMAP2 = False
    tSNE = False
    TopoAE = False

    UMAP_final = False
    tSNE_final = False

    WCAE = True

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
    elif tSNE_final:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/tsne_final'
        root_save = '/Users/simons/MT_data/eval_all_analysis/Competitors/tSNE/FINAL'
    elif WCAE:
        exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise'
        df_path = '/Users/simons/PycharmProjects/MT-VAEs-TDA/output/eval/WCAE/metrics_selected_processed_new_new.csv'
        root_save = '/Users/simons/MT_data/eval_all_analysis/WCAE/SwissRoll_nonoise'
    else:
        ValueError


    print(exp_dir)
    if WCAE:
        df_path = df_path
    else:
        df_path = os.path.join(exp_dir, 'eval_metrics_all.csv')
    print('Load data...')
    df = pd.read_csv(df_path)

    bss = [64, 128, 256, 512]

    metrics_min = ['rmse_manifold_Z','test_mean_Lipschitz_std_refZ', 'test_mean_mrre',
                   'training.metrics.notmatched_pairs_0D', 'test_rmse', 'test_llrmse_X_norm15',
                   'test_llrmse_X_norm5', 'test_density_kl_global_00001', 'test_density_kl_global_001', 'test_density_kl_global_01']
    metrics_max = ['test_mean_continuity', 'test_mean_trustworthiness',
                   'training.metrics.matched_pairs_0D']

    dir_name_mapping = {
        'rmse_manifold_Z' : 'rmse_manifold_Z',
        'training.metrics.notmatched_pairs_0D': 'nonmatchingpairs',
        'test_rmse'                           : 'rmse',
        'test_llrmse_X_norm15'                : 'llrmse_norm15',
        'test_llrmse_X_norm5'                 : 'llrmse_norm5',
        'test_density_kl_global_001'          : 'kl_global_001',
        'test_density_kl_global_00001': 'kl_global_00001',
        'test_density_kl_global_01'           : 'kl_global_01',
        'test_mean_continuity'                : 'mean_continuity',
        'test_mean_trustworthiness'           : 'mean_trustworthiness',
        'test_mean_mrre'                      : 'mean_mrre',
        'training.metrics.matched_pairs_0D'   : 'matchedpairs',
        'test_mean_Lipschitz_std_refZ'        : 'lipschitz_std'
    }

    metrics_min_comp = ['rmse_manifold_Z','train_mean_mrre', 'train_rmse',
                        'train_llrmse_X_norm15', 'train_llrmse_X_norm5',
                        'train_density_kl_global_001','train_density_kl_global_00001',
                        'train_density_kl_global_01', 'test_mean_Lipschitz_std_refZ']
    metrics_max_comp = ['train_mean_continuity', 'train_mean_trustworthiness']
    dir_name_mapping_comp = {
        'rmse_manifold_Z' : 'rmse_manifold_Z',
        'train_rmse'                  : 'rmse',
        'train_llrmse_X_norm15'       : 'llrmse_norm15',
        'train_llrmse_X_norm5'        : 'llrmse_norm5',
        'train_density_kl_global_00001': 'kl_global_00001',
        'train_density_kl_global_001' : 'kl_global_001',
        'train_density_kl_global_01'  : 'kl_global_01',
        'train_mean_continuity'       : 'mean_continuity',
        'train_mean_trustworthiness'  : 'mean_trustworthiness',
        'train_mean_mrre'             : 'mean_mrre',
        'test_mean_Lipschitz_std_refZ': 'lipschitz_std'
    }

    if UMAP or tSNE or UMAP2 or UMAP_final or tSNE_final:
        bss = ['const']
        metrics_min = metrics_min_comp
        metrics_max = metrics_max_comp
        dir_name_mapping = dir_name_mapping_comp

    for bs in bss:
        if UMAP or tSNE or UMAP2 or UMAP_final or tSNE_final:
            df_temp = df
        else:
            df_temp = df[df.batch_size == bs]

        for metric in (metrics_max + metrics_min):

            df_temp_metric = df_temp[df_temp['metric'] == metric]

            df_temp_metric = df_temp_metric[['uid','value','seed']]
            # select top N IDs create like a dict that maps rank to IDs
            if metric in metrics_max:
                df_selected = df_temp_metric.sort_values('value',ascending = False).groupby('seed').head(N)

            else:
                df_selected = df_temp_metric.sort_values('value', ascending=True).groupby(
                    'seed').head(N)

            uids = list(df_selected.uid)

            uid_dict = dict()
            rank_count = 1
            for uid in uids:
                uid_dict.update({uid: '{}_{}'.format(rank_count, uid)})
                rank_count += 1

            # create directory for selection
            eval_root = os.path.join(root_save, 'bs{}_{}'.format(bs, dir_name_mapping[metric]))
            try:
                os.makedirs(eval_root)
            except:
                pass

            # get latents and save in directory
            rank_count = 1
            for uid in uids:
                latent_name = '{}_{}'.format(rank_count, uid)
                get_latent_rename(exp_dir, eval_root, uid, latent_name)
                rank_count += 1

            # save df_metric of selection
            df_selected.to_csv(
                os.path.join(eval_root, 'metrics_selected_{}.csv'.format(dir_name_mapping[metric])))
