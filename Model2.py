import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm

columnName = ['seed', 'lat', 'long'	'fg2pct', 'fg3pct', 'ftpct', 'blockpct', 'oppfg2pct', 'oppfg3pct', 'oppftpct',
              'oppblockpct', 'f3grate', 'oppf3grate', 'arate', 'opparate', 'stlrate', 'oppstlrate', 'tempo', 'adjtempo',
              'oe', 'adjoe', 'de', 'adjde', 'rpi_rating', 'pt_school_ncaa', 'pt_overall_ncaa', 'pt_school_s16', 'pt_overall_s16',
              'pt_school_ff', 'pt_overall_ff', 'pt_career_school_wins',	'pt_career_school_losses', 'pt_career_overall_losses',
              'team_season_wins', 'pt_team_season_losses', 'pt_coach_season_wins', 'pt_coach_season_losses']


import numpy as np

df_all = pd.read_csv('NCAA_Tourney_2002-2016_1.csv')
df = df_all.iloc[:, :-4]
df_out = df_all.iloc[:, -4:]

df_team1 = df_all.iloc[:, 2:38]
df_team1.columns = columnName
df_team2 = df_all.iloc[:, 39:75]
df_team2.columns = columnName


df_team1.loc[:, 'expwin'] = df.loc[:, 'team1_adjoe'] ** 11.5 / (df.loc[:, 'team1_adjde'] ** 11.5
                                                            + df.loc[:, 'team1_adjoe'] ** 11.5)


df_team2.loc[:, 'expwin'] = df.loc[:, 'team2_adjoe'] ** 11.5 / (df.loc[:, 'team2_adjde'] ** 11.5
                                                            + df.loc[:, 'team2_adjoe'] ** 11.5)


df_team1.loc[:, 'log5'] = (df_team1.loc[:, 'expwin'] - df_team1.loc[:, 'expwin'] * df_team2.loc[:, 'expwin']) / \
            ((df_team1.loc[:, 'expwin'] + df_team2.loc[:, 'expwin'] - 2* df_team1.loc[:, 'expwin'] * df_team2.loc[:, 'expwin']))

df_team2.loc[:, 'log5'] = (df_team2.loc[:, 'expwin'] - df_team1.loc[:, 'expwin'] * df_team2.loc[:, 'expwin']) / \
            ((df_team1.loc[:, 'expwin'] + df_team2.loc[:, 'expwin'] - 2* df_team1.loc[:, 'expwin'] * df_team2.loc[:, 'expwin']))

data = df_team1 - df_team2


normolizer = preprocessing.Normalizer().fit(data)
df_norm = normolizer.transform(data)

pca = PCA(svd_solver='full')
pca.fit(df_norm)
variance = pca.explained_variance_
cumulativeExplainedVarianceRatio = np.cumsum(variance)/np.sum(variance)

PCA_matrix = pd.DataFrame(pca.components_, index=data.columns)


PCA_matrix.to_csv('pca_result2.csv')



newData = pd.DataFrame(pca.transform(df_norm))
newData = newData.loc[:, 0:15]

X = newData
y = df_out.loc[:, 'result']
kf = KFold(n_splits=4, shuffle=False)

for train, test in kf.split(X):

    X_train, X_valid, y_train, y_valid = train_test_split(X.loc[train], y.loc[train], test_size=0.2)

    X_test, y_test = X.loc[test], y.loc[test]

    paramsRF = {'n_estimators': 100, 'min_samples_split': 5}

    #clf = MLPClassifier(alpha=1e-5, learning_rate='constant', momentum=0.9, hidden_layer_sizes=(50,), max_iter=10
    #                   , verbose=10, tol=1e-4, random_state=1, activation='logistic', solver='adam')

    #clf = ensemble.RandomForestClassifier(**paramsRF)

    clf = linear_model.LogisticRegressionCV(Cs=4, solver='liblinear', max_iter=1000, tol=1e-5, scoring='neg_log_loss')
    #clf = GaussianNB()
    #clf = linear_model.ElasticNetCV(l1_ratio=0)
    #clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    #clf=linear_model.LassoLarsIC(criterion='aic')

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv='prefit')
    sig_clf.fit(X_valid, y_valid)

    df_out.loc[test, 'prob1'] = sig_clf.predict_proba(X_test)[:, 0]
    df_out.loc[test, 'prob2'] = sig_clf.predict_proba(X_test)[:, 1]

df_out.loc[:, 'log_loss1'] = df_out.loc[:, 'result'] * np.log(df_out.loc[:, 'prob1']) \
                             + (1 - df_out.loc[:, 'result']) * np.log((1 - df_out.loc[:, 'prob1']))

df_out.loc[:, 'log_loss2'] = df_out.loc[:, 'result'] * np.log(df_out.loc[:, 'prob2']) \
                             + (1 - df_out.loc[:, 'result']) * np.log((1 - df_out.loc[:, 'prob2']))

log_loss_1 = - df_out.loc[:, 'log_loss1'].sum()/ len(df_out)
log_loss_2 = - df_out.loc[:, 'log_loss2'].sum()/ len(df_out)

print(log_loss_1, log_loss_2)
#df_out.to_csv('output.csv')
