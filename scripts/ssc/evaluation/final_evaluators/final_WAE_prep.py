import os

import pandas as pd



root_path1 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise2'
root_path2 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise'
root_path3 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/WCAE_swissroll_nonoise2_PREV'

if __name__ == "__main__":

    df1 = pd.read_csv(os.path.join(root_path1, 'eval_metrics_all.csv'))
    df2 = pd.read_csv(os.path.join(root_path2, 'eval_metrics_all.csv'))
    df3 = pd.read_csv(os.path.join(root_path3, 'eval_metrics_all.csv'))

    uid1 = list(set(list(df1.uid)))
    uid2 = list(set(list(df2.uid)))
    uid3 = list(set(list(df3.uid)))


    seed1 = []
    for uid in uid1:
        seed1.append(uid.split('-')[2][4:])
    seed1 = list(set(seed1))
    print('DONE 1! - {}'.format(seed1))
    seed2 = []
    for uid in uid2:
        seed2.append(uid.split('-')[2][4:])
    seed2 = list(set(seed2))
    print('DONE 2! - {}'.format(seed2))
    seed3 = []
    for uid in uid3:
        seed3.append(uid.split('-')[2][4:])
    seed3 = list(set(seed3))

    print('SEED1: {}'.format(seed1))
    print('SEED2: {}'.format(seed2))
    print('SEED3: {}'.format(seed3))

