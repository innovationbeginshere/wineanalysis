Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> import seaborn as sns
>>> import matplotlib.pyplot as plt
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import SVC
>>> from sklearn import svm
>>> from sklearn.metrics import confusion_matrix,classification_report
>>> from sklearn.preprocessing import StandardScaler,LabelEncoder
>>> from sklearn.model_selection import train_test_split
>>> wine=pd.read_csv('wine.csv')
>>> wine.head()
   fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.4              0.70         0.00  ...       0.56      9.4        5
1            7.8              0.88         0.00  ...       0.68      9.8        5
2            7.8              0.76         0.04  ...       0.65      9.8        5
3           11.2              0.28         0.56  ...       0.58      9.8        6
4            7.4              0.70         0.00  ...       0.56      9.4        5

[5 rows x 12 columns]
>>> wine.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
fixed acidity           1599 non-null float64
volatile acidity        1599 non-null float64
citric acid             1599 non-null float64
residual sugar          1599 non-null float64
chlorides               1599 non-null float64
free sulfur dioxide     1599 non-null float64
total sulfur dioxide    1599 non-null float64
density                 1599 non-null float64
pH                      1599 non-null float64
sulphates               1599 non-null float64
alcohol                 1599 non-null float64
quality                 1599 non-null int64
dtypes: float64(11), int64(1)
memory usage: 150.0 KB
>>> wine.isnull().sum()
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64
>>> bins=(2,6.5,8)   #2 for in how much types it should be classified ,6.5to find quality,if value<6.5 bad wine,if >6.5 good wine
>>> group_name=['bad','good']
>>> wine['quality']=pd.cut(wine['quality'],bins=bins,labels=group_name)
>>> wine['quality'].unique()
[bad, good]
Categories (2, object): [bad < good]
>>> label_quality=LabelEncoder()      #encoder :we got values in good and bd to transform it to 1 & 0
>>> wine['quality']=label_quality.fit_transform(wine['quality'])
>>> wine.head()
   fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.4              0.70         0.00  ...       0.56      9.4        0
1            7.8              0.88         0.00  ...       0.68      9.8        0
2            7.8              0.76         0.04  ...       0.65      9.8        0
3           11.2              0.28         0.56  ...       0.58      9.8        0
4            7.4              0.70         0.00  ...       0.56      9.4        0

[5 rows x 12 columns]
>>> wine.head(10)
   fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.4              0.70         0.00  ...       0.56      9.4        0
1            7.8              0.88         0.00  ...       0.68      9.8        0
2            7.8              0.76         0.04  ...       0.65      9.8        0
3           11.2              0.28         0.56  ...       0.58      9.8        0
4            7.4              0.70         0.00  ...       0.56      9.4        0
5            7.4              0.66         0.00  ...       0.56      9.4        0
6            7.9              0.60         0.06  ...       0.46      9.4        0
7            7.3              0.65         0.00  ...       0.47     10.0        1
8            7.8              0.58         0.02  ...       0.57      9.5        1
9            7.5              0.50         0.36  ...       0.80     10.5        0

[10 rows x 12 columns]
>>> wine['quality'].value_counts()
0    1382
1     217
Name: quality, dtype: int64
>>> sns.countplot(wine['quality'])
<matplotlib.axes._subplots.AxesSubplot object at 0x00000230170FACC0>
>>> plt.show()


>>> X=wine.drop('quality',axis=1)   #features minus what we are looking for 
>>> y=wine['quality']               #features we are looking for
>>> #training of data
>>> X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
>>> sc=StandardScaler()
>>> X_train=sc.fit_transform(X_train)   #compare the data to actual data and then transform it
>>> X_test=sc.fit_transform(X_test)
>>> X_train[:10]
array([[ 0.21833164,  0.88971201,  0.19209222,  0.30972563, -0.04964208,
         0.69100692,  1.04293362,  1.84669643,  1.09349989,  0.45822284,
         1.12317723],
       [-1.29016623, -1.78878251,  0.65275338, -0.80507963, -0.45521361,
         2.38847304,  3.59387025, -3.00449133, -0.40043872, -0.40119696,
         1.40827174],
       [ 1.49475291, -0.78434707,  1.01104539, -0.52637831,  0.59927236,
        -0.95796016, -0.99174203,  0.76865471, -0.07566946,  0.51551749,
        -0.58738978],
       [ 0.27635078,  0.86181102, -0.06383064, -0.66572897, -0.00908493,
         0.01202048, -0.71842739,  0.08948842,  0.05423824, -1.08873281,
        -0.96751578],
       [ 0.04427419,  2.81487994, -0.62686095,  2.39998549, -0.31326357,
        -0.47296984,  0.2229897 ,  1.1998714 ,  0.37900751, -0.9741435 ,
        -0.49235828],
       [-0.07176411, -0.78434707,  1.11341454, -0.17800167,  0.21397941,
         3.01896045,  2.62208486,  0.60694845,  0.44396136,  1.89058918,
        -0.58738978],
       [-1.17412793,  0.10848444, -0.62686095, -0.52637831, -0.23214927,
         0.98200112, -0.35400787, -1.95879086,  0.05423824,  0.91658007,
         1.12317723],
       [-0.1878024 , -0.17052541,  0.60156881,  0.03102432, -0.13075639,
        -0.37597178, -0.01995665,  0.93036097,  0.76873063, -0.229313  ,
         0.26789373],
       [-0.07176411,  0.61070216, -0.01264607, -0.38702766,  0.13286511,
        -1.05495822,  0.92146044,  0.37516948, -1.17988496, -0.229313  ,
        -1.25261029],
       [ 1.8428678 , -1.95618842,  1.21578369,  1.00647892,  0.31537229,
        -1.15195628, -0.71842739,  1.52328391, -0.20557717,  1.77599987,
        -0.30229528]])
>>>#Random forest model
>>> rfc=RandomForestClassifier(n_estimators=200)
>>> rfc.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
>>> pred_rfc=rfc.predict(X_test)
>>> pred_rfc[:20]
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
>>> print(classification_report(y_test,pred_rfc))#how this model performed
              precision    recall  f1-score   support

           0       0.90      0.97      0.93       273
           1       0.71      0.36      0.48        47

    accuracy                           0.88       320
   macro avg       0.80      0.67      0.71       320
weighted avg       0.87      0.88      0.87       320

>>> print(confusion_matrix(y_test,pred_rfc))
[[266   7]  #266 prediction correct for bad wines and 7 prediction wrong 
 [ 30  17]] #30 prediction correct for good wines and 17 prediction wrong
>>> #SVM model
>>> svm=svm.SVC()
>>> svm.fit(X_train,y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
>>> pred_svm=svm.predict(X_test)
>>> print(classification_report(y_test,pred_svm))
              precision    recall  f1-score   support

           0       0.88      0.98      0.93       273
           1       0.71      0.26      0.37        47

    accuracy                           0.88       320
   macro avg       0.80      0.62      0.65       320
weighted avg       0.86      0.88      0.85       320

>>> print(confusion_matrix(y_test,pred_svm))
[[268   5]  #268 prediction correct for bad wines and 5 prediction wrong 
 [ 35  12]] #35 prediction correct for good wines and 12 prediction wrong
>>> 
