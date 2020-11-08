import argparse
import os
from distutils.dir_util import copy_tree

import pandas as pd
import shutil


def get_plot_rename(exp_root, eval_root, uid, latent_name, plot='train_latent_visualization'):
    uid_rood = os.path.join(exp_root, uid)

    existing_file = open(os.path.join(uid_rood, '{}.pdf'.format(plot)), "r")
    new_file = open(os.path.join(eval_root, latent_name+'.pdf'), "w")

    src_file = os.path.join(uid_rood, '{}.pdf'.format(plot))
    dest_dir = eval_root

    shutil.copy(src_file, dest_dir)  # copy the file to destination dir

    dst_file = os.path.join(eval_root, '{}.pdf'.format(plot))
    new_dst_file_name = os.path.join(eval_root, latent_name+'.pdf')

    os.rename(dst_file, new_dst_file_name)  # rename
    os.chdir(dest_dir)


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--directory", help="Experiment directory", type=str)
    parser.add_argument('-n', "--n", default=1, help="Numbber of topresults ", type=int)
    parser.add_argument('--competitor', help='Model is a competitor', action='store_true')
    parser.add_argument('--noseed',
                        help='Indicate that seed column is not avail in eval_metrics_all.csv',
                        action='store_true')
    parser.add_argument('--train_latent', help='Get train latent', action='store_true')
    parser.add_argument('--test_latent', help='Get test latent', action='store_true')
    parser.add_argument('--manifold', help='Get manifold', action='store_true')
    parser.add_argument('--exp', help='Get all experiment data of selected', action='store_true')


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    # SET DF PATH
    exp_dir = args.directory
    N = args.n
    if args.competitor:
        metrics_to_select = ['rmse_manifold_Z', 'train_mean_Lipschitz_std_refZ','training.metrics.notmatched_pairs_0D','training.loss.autoencoder','train_continuity']
        max_metrics = ['train_continuity']
    else:
        metrics_to_select = ['rmse_manifold_Z', 'test_mean_Lipschitz_std_refZ',
                             'training.metrics.notmatched_pairs_0D', 'training.loss.autoencoder',
                             'test_continuity']
        max_metrics = ['test_continuity']

    # LOAD DF
    df = pd.read_csv(os.path.join(exp_dir, 'eval_metrics_all.csv'))

    for metric in metrics_to_select:
        df_selected = df[df.metric == metric]
        # get column with seed
        if args.noseed:
            df_selected['seed'] = 0
            for uuid in list(set(list(df_selected.uid))):
                if args.competitor:
                    df_selected.loc[(df_selected.uid == uuid), ['seed']] = int(
                        uuid.split('-')[6][4:])
                else:
                    df_selected.loc[(df_selected.uid == uuid), ['seed']] = int(
                        uuid.split('-')[10][4:])
        else:
            pass

        if args.competitor:
            df_selected = df_selected[['uid', 'seed', 'value']]
            if metric in max_metrics:
                df_selected = df_selected.sort_values('value', ascending=False).groupby(
                    ['seed']).head(N)
            else:
                df_selected = df_selected.sort_values('value', ascending=True).groupby(
                    ['seed']).head(N)
            bss = ['na']
        else:
            df_selected = df_selected[['uid', 'seed', 'batch_size', 'value']]
            if metric in max_metrics:
                df_selected = df_selected.sort_values('value', ascending=False).groupby(
                    ['seed', 'batch_size']).head(N)
            else:
                df_selected = df_selected.sort_values('value', ascending=True).groupby(
                ['seed', 'batch_size']).head(N)
            bss = list(set(list(df_selected['batch_size'])))

        for bs in bss:
            # create directory for selection
            eval_root = os.path.join(exp_dir, 'selector{}'.format(N),
                                     'bs{}_metric{}'.format(bs, metric))
            try:
                os.makedirs(eval_root)
            except:
                pass

            if args.competitor:
                uids = list(df_selected.uid)
            else:
                df_selected_bs = df_selected[df_selected['batch_size'] == bs]
                uids = list(df_selected_bs.uid)
            uid_dict = dict()
            rank_count = 1
            for uid in uids:
                uid_dict.update({uid: '{}_{}'.format(rank_count, uid)})
                train_latent_name = 'train_latents{}_{}'.format(rank_count, uid)
                test_latent_name = 'test_latents{}_{}'.format(rank_count, uid)
                manifolddist_name = 'manifold_dist{}_{}'.format(rank_count, uid)

                if args.train_latent:
                    get_plot_rename(exp_dir, eval_root, uid, train_latent_name)
                if args.test_latent:
                    get_plot_rename(exp_dir, eval_root, uid, test_latent_name,
                                    plot='test_latent_visualization')
                if args.manifold:
                    get_plot_rename(exp_dir, eval_root, uid, manifolddist_name,
                                    plot='manifold_Z_distcomp')
                if args.exp:
                    copy_tree(os.path.join(exp_dir, uid), os.path.join(eval_root, uid))
                rank_count += 1

            df_selected.to_csv(
                os.path.join(eval_root, 'metrics_selected.csv'))

        # df_selected[['uid','batch_size','seed','value']].groupby(['batch_size','seed'], as_index=False).min()
