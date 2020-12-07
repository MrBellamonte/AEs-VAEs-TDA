import os

import pandas as pd



if __name__ == "__main__":

    df_path_topoae1 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_topoae_1'
    df_path_topoae_deep1 = '/Users/simons/MT_data/sync/euler_sync_scratch/schsimo/output/mnist_topoae_1_deepae'

    df_path = df_path_topoae_deep1

    df = pd.read_csv(os.path.join(df_path,'eval_metrics_all.csv'))

    df['bs'] = 0
    df['seed'] = 0

    uids = list(set(df.uid.values))

    for uid in uids:
        df.loc[df['uid'] == uid,'bs'] = uid.split('-')[5][2:]
        df.loc[df['uid'] == uid, 'seed'] = uid.split('-')[9][4:]


    df.to_csv(os.path.join(df_path,'eval_metrics_all_2.csv'))