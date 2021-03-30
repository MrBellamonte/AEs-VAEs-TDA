import os
import random

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

if __name__ == "__main__":

    path = '/Users/simons/MT_data/eval_data/MNIST3D_FINAL/latents'

    name = 'FINAL_SVM_kNN'

    exp_topoae_iso = 'topoae_path_iso'
    exp_ae_iso = 'ae_path_iso'

    exp_topoae_kl01 = 'topoae_path_kl01'
    exp_ae_kl01 = 'ae_path_kl01'

    exp_wae_kl01 = 'wae_path_kl01'
    exp_wae_iso = 'wae_path_stdiso'

    exp_umap_iso = 'umap_path_iso'
    exp_umap_kl = 'umap_path_kl'
    exp_tsne_iso = 'tsne_path_iso'
    exp_tsne_kl = 'tsne_path_kl'


    df_final = pd.DataFrame()
    for exp in [exp_topoae_iso, exp_ae_iso, exp_topoae_kl01, exp_ae_kl01,exp_wae_kl01, exp_wae_iso, exp_umap_iso, exp_umap_kl, exp_tsne_iso, exp_tsne_kl]:
        ii = 0
        print(exp)

        while(ii<2):
            latents = np.load(os.path.join(path,'{}.npy'.format(exp_ae_iso)))
            labels = np.load(os.path.join(path,'labels{}.npy'.format(exp_ae_iso)))


            ind_mixed = random.sample(range(latents.shape[0]), latents.shape[0])

            ind_train = ind_mixed[:7500]
            ind_test = ind_mixed[7500:]

            data_train = latents[ind_train]
            labels_train = labels[ind_train]

            data_test = latents[ind_test]
            labels_test = labels[ind_test]

            scaler = StandardScaler()
            scaler.fit(X = data_train)

            data_train = scaler.transform(data_train)
            data_test = scaler.transform(data_test)



            if ii == 0:
                print('kNN!')
                model = KNeighborsClassifier(n_jobs=4)
                model.fit(X = data_train,y = labels_train)
                y_pred = model.predict(data_test)
                acc = accuracy_score(labels_test, y_pred)



                df_temp = pd.DataFrame(index=np.arange(1), columns=['method','acc'])

                df_temp['method'] = 'kNN'
                df_temp['model'] = exp
                df_temp['acc'] = acc


                df_final = df_final.append(df_temp)
            elif ii == 1:
                print('SVM!')
                parameters = {'C': np.logspace(-3,1,5).tolist(),'max_iter' : np.array([50000])}
                model = LinearSVC()
                clf = GridSearchCV(model, parameters,n_jobs=5)
                clf.fit(X = data_train, y = labels_train)

                #print(clf.best_params_)
                svc_final = LinearSVC(**clf.best_params_)
                svc_final.fit(X = data_train,y = labels_train)
                y_pred = svc_final.predict(data_test)
                acc = accuracy_score(labels_test, y_pred)

                df_temp = pd.DataFrame(index=np.arange(1), columns=['method', 'acc'])

                df_temp['method'] = 'SVM'
                df_temp['model'] = exp
                df_temp['acc'] = acc

                df_final = df_final.append(df_temp)
            else:
                pass

            ii = ii + 1

    df_final.to_csv(os.path.join('/Users/simons/MT_data/eval_data/MNIST3D_FINAL','{}.csv'.format(name)))
    
