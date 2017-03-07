import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct


df_all = pd.read_csv('NCAA_Tourney_2002-2016_2.csv')

df = df_all.iloc[:, :-4]
df_out = df_all.iloc[:, -4:]

df2 = pd.DataFrame()

df2.loc[:, 'seed_distance'] = df.loc[:, 'team1_seed'] - df.loc[:, 'team2_seed']
df2.loc[:, 'rpi_distance'] = df.loc[:, 'team1_rpi_rating'] - df.loc[:, 'team2_rpi_rating']
df2.loc[:, 'stlrate_distance'] = df.loc[:, 'team1_stlrate'] - df.loc[:, 'team2_stlrate']

df2.loc[:, 'expwin1'] = df.loc[:, 'team1_adjoe'] ** 11.5 / (df.loc[:, 'team1_adjde'] ** 11.5
                                                            + df.loc[:, 'team1_adjoe'] ** 11.5)


df2.loc[:, 'expwin2'] = df.loc[:, 'team2_adjoe'] ** 11.5 / (df.loc[:, 'team2_adjde'] ** 11.5
                                                            + df.loc[:, 'team2_adjoe'] ** 11.5)


df2.loc[:, 'team1log5'] = (df2.loc[:, 'expwin1'] - df2.loc[:, 'expwin1'] * df2.loc[:, 'expwin2']) / \
            ((df2.loc[:, 'expwin1'] + df2.loc[:, 'expwin2'] - 2* df2.loc[:, 'expwin1'] * df2.loc[:, 'expwin2']))

df2.loc[:, 'team2log5'] = (df2.loc[:, 'expwin2'] - df2.loc[:, 'expwin1'] * df2.loc[:, 'expwin2']) / \
            ((df2.loc[:, 'expwin1'] + df2.loc[:, 'expwin2'] - 2* df2.loc[:, 'expwin1'] * df2.loc[:, 'expwin2']))

df2.loc[:, 'prob_diff'] = df2.loc[:, 'team1log5'] - df2.loc[:, 'team2log5']

#df3 = df2.loc[:, ['seed_distance', 'rpi_distance', 'stlrate_distance', 'prob_diff']]


#normolizer = preprocessing.Normalizer().fit(df3)
#df_norm = normolizer.transform(df3)

#pca = PCA(svd_solver='full')
#pca = KernelPCA()
#pca.fit(df_norm)
#variance = pca.explained_variance_
#cumulativeExplainedVarianceRatio = np.cumsum(variance)/np.sum(variance)

#PCA_matrix = pd.DataFrame(pca.components_, index=df.columns).loc[:, 0:27]

#newData = pd.DataFrame(pca.transform(df_norm))
#newData = newData.loc[:, 0:20]

newData = df2.loc[:, ['seed_distance', 'rpi_distance', 'stlrate_distance', 'prob_diff']]

X = newData
y = df_out.loc[:, 'result']
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
X_valid = pd.DataFrame()
y_valid = pd.DataFrame()


for year in range(2002, 2010):
    X_train = pd.concat([X_train, X.loc[df_out.Season == year]])
    y_train = pd.concat([y_train, y.loc[df_out.Season == year]])


for year in range(2010, 2014):
    X_valid = pd.concat([X_valid, X.loc[df_out.Season == year]])
    y_valid = pd.concat([y_valid, y.loc[df_out.Season == year]])


for year in range(2014, 2017):
    X_test = pd.concat([X_test, X.loc[df_out.Season == year]])
    y_test = pd.concat([y_test, y.loc[df_out.Season == year]])


#clf = MLPClassifier(alpha=1e-15, learning_rate='constant', momentum=0, hidden_layer_sizes=(100,5), max_iter=200
#                    , tol=10, random_state=1, activation='logistic', solver='adam')

clf = linear_model.LogisticRegressionCV(Cs=2, solver='liblinear', max_iter=500, tol=1e-6, scoring='neg_log_loss')
#clf = linear_model.ElasticNetCV()
#clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
#clf=linear_model.LassoLarsIC()

clf.fit(X_train, y_train.values.ravel())

sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=10)
sig_clf.fit(X_valid, y_valid.values.ravel())


y_pred = sig_clf.predict_proba(X_test)
logLoss = log_loss(y_true=y_test, y_pred=y_pred)

print(logLoss)