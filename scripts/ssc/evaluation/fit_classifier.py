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

    name = 'kNN'

    exp_topoae_iso = 'topoae_path_iso'
    exp_ae_iso = 'ae_path_iso'
    exp_topoae_kl01 = 'topoae_path_kl01'
    exp_ae_kl01 = 'ae_path_kl01'

    df_final = pd.DataFrame()
    for exp in [exp_topoae_iso, exp_ae_iso, exp_topoae_kl01, exp_ae_kl01]:
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



        # parameters = {'C': np.logspace(-3,1,5).tolist(),'max_iter' : np.array([50000])}
        # model = LinearSVC()
        # clf = GridSearchCV(model, parameters,n_jobs=5)
        # clf.fit(X = data_train, y = labels_train)
        # 
        # print(clf.best_params_)
        # svc_final = LinearSVC(**clf.best_params_)
        # svc_final.fit(X = data_train,y = labels_train)
        # y_pred = svc_final.predict(data_test)
        # acc = accuracy_score(labels_test, y_pred)
        
        
        model = KNeighborsClassifier(n_jobs=4)
        model.fit(X = data_train,y = labels_train)
        y_pred = model.predict(data_test)
        acc = accuracy_score(labels_test, y_pred)
        
        
        
        df_temp = pd.DataFrame(index=np.arange(1), columns=['method','acc'])

        df_temp['model'] = exp
        df_temp['acc'] = acc

        df_final = df_final.append(df_temp)


    df_final.to_csv(os.path.join('/Users/simons/MT_data/eval_data/MNIST3D_FINAL','{}.csv'.format(name)))
    
