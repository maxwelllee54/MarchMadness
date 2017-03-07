import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm


import numpy as np

df_all = pd.read_csv('NCAA_Tourney_2002-2016_1.csv')
df = df_all.iloc[:, :-4]
df_out = df_all.iloc[:, -4:]


normolizer = preprocessing.Normalizer().fit(df)
df_norm = normolizer.transform(df)

pca = PCA(svd_solver='full')
pca.fit(df_norm)
variance = pca.explained_variance_
cumulativeExplainedVarianceRatio = np.cumsum(variance)/np.sum(variance)

PCA_matrix = pd.DataFrame(pca.components_, index=df.columns).loc[:, 0:20]

newData = pd.DataFrame(pca.transform(df_norm))
newData = newData.loc[:, 0:20]

X = newData
y = df_out.loc[:, 'result']
kf = KFold(n_splits=5, shuffle=True)

for train, test in kf.split(X):

    X_train, X_valid, y_train, y_valid = train_test_split(X.loc[train], y.loc[train], test_size=0.2, random_state=42)

    X_test, y_test = X.loc[test], y.loc[test]

    paramsRF = {'n_estimators': 100, 'min_samples_split': 5}
    clf = svm.SVR()

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv='prefit')
    sig_clf.fit(X_valid, y_valid)

    df_out.loc[test, 'prob1'] = sig_clf.predict(X_test)
    df_out.loc[test, 'prob2'] = sig_clf.predict(X_test)


df_out.loc[:, 'log_loss1'] = df_out.loc[:, 'result'] * np.log(df_out.loc[:, 'prob1']) \
                             + (1 - df_out.loc[:, 'result']) * np.log((1 - df_out.loc[:, 'prob1']))

df_out.loc[:, 'log_loss2'] = df_out.loc[:, 'result'] * np.log(df_out.loc[:, 'prob2']) \
                             + (1 - df_out.loc[:, 'result']) * np.log((1 - df_out.loc[:, 'prob2']))

log_loss_1 = - df_out.loc[:, 'log_loss1'].sum()/ len(df_out)
log_loss_2 = - df_out.loc[:, 'log_loss2'].sum()/ len(df_out)

print(log_loss_1, log_loss_2)
df_out.to_csv('output2.csv')