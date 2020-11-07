import os
import time
from fractions import Fraction

import pandas as pd


if __name__ == "__main__":

    N = 3

    bss = [64, 128, 256, 512]

    exp_dir = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise_FINAL'
    root_save = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise_FINAL'
    df_path = os.path.join(exp_dir, 'eval_metrics_all.csv')

    criterion = 'test_mean_Lipschitz_std_refZ'
    max_metrics = ['test_mean_trustworthiness','test_mean_continuity']



    print('Load data...')
    df = pd.read_csv(df_path)

    df['seed'] = 0
    df['k'] = 0
    df['mu_push'] = 0
    lenght = len(list(set(list(df.uid))))
    for i,uuid in enumerate(list(set(list(df.uid)))):
        df.loc[(df.uid == uuid), ['seed']] = int(uuid.split('-')[10][4:])
        df.loc[(df.uid == uuid), ['k']] = int(
            uuid.split('-')[12][1:])
        df.loc[(df.uid == uuid), ['mu_push']] = float(Fraction(uuid.split('-')[11][13:].replace('_', '/')))
        if (i+1) % 100==0:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            print('{} out of {}'.format((i+1),lenght))


    df.to_csv(os.path.join(root_save,'eval_metrics_all_wkmu.csv'))

