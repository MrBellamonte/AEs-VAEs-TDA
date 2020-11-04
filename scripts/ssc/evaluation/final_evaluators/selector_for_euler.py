import argparse
import os

import pandas as pd
import shutil

def get_plot_rename(exp_root, eval_root, uid, latent_name, plot = 'train_latent_visualization'):
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
    parser.add_argument('-dir', "--directory",

                        help="Experiment directory", type=str)
    parser.add_argument('-n', "--n",
                        default = 1,
                        help="Numbber of topresults ", type=int)
    parser.add_argument('-c', "--competitor",
                        default = False,
                        help="Numbber of topresults ", type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_input()

    # SET DF PATH
    exp_dir = args.directory
    competitor = args.competitor
    N = args.n

    metrics_to_select = ['rmse_manifold_Z', 'test_mean_Lipschitz_std_refZ']

    # LOAD DF
    df = pd.read_csv(os.path.join(exp_dir,'eval_metrics_all.csv'))



    for metric in metrics_to_select:
        df_selected = df[df.metric == metric]
        # get column with seed
        df_selected['seed'] = 0

        for uuid in list(set(list(df_selected.uid))):

            if competitor:
                df_selected.loc[(df_selected.uid == uuid), ['seed']] = int(uuid.split('-')[6][4:])
            else:
                df_selected.loc[(df_selected.uid == uuid),['seed']] = int(uuid.split('-')[10][4:])

        if competitor:
            df_selected = df_selected[['uid','seed','value']]
            df_selected = df_selected.sort_values('value', ascending=True).groupby(
                ['seed']).head(N)
            bss = ['na']
        else:
            df_selected = df_selected[['uid','seed','batch_size','value']]
            df_selected = df_selected.sort_values('value', ascending=True).groupby(
                ['seed','batch_size']).head(N)
            bss = list(set(list(df_selected['batch_size'])))



        for bs in bss:
            # create directory for selection
            eval_root = os.path.join(exp_dir,'selector{}'.format(N), 'bs{}_metric{}'.format(bs,metric))
            try:
                os.makedirs(eval_root)
            except:
                pass

            if competitor:
                pass
            else:
                df_selected_bs = df_selected[df_selected['batch_size']==bs]
            uids = list(df_selected_bs.uid)
            uid_dict = dict()
            rank_count = 1
            for uid in uids:
                uid_dict.update({uid: '{}_{}'.format(rank_count, uid)})
                latent_name = 'latents{}_{}'.format(rank_count, uid)
                manifolddist_name = 'manifold_dist{}_{}'.format(rank_count, uid)


                get_plot_rename(exp_dir, eval_root, uid, latent_name)
                get_plot_rename(exp_dir, eval_root, uid, manifolddist_name,plot = 'manifold_Z_distcomp')
                rank_count += 1

            df_selected.to_csv(
                os.path.join(eval_root, 'metrics_selected.csv'))




        #df_selected[['uid','batch_size','seed','value']].groupby(['batch_size','seed'], as_index=False).min()

