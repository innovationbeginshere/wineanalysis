
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
wine=pd.read_csv('wine.csv')
wine.head()
wine.info()
wine.isnull().sum()

bins=(2,6.5,8)   #2 for in how much types it should be classified ,6.5to find quality,if value<6.5 bad wine,if >6.5 good wine
group_name=['bad','good']
wine['quality']=pd.cut(wine['quality'],bins=bins,labels=group_name)
wine['quality'].unique()
label_quality=LabelEncoder()      #encoder :we got values in good and bd to transform it to 1 & 0
wine['quality']=label_quality.fit_transform(wine['quality'])
wine.head()
wine.head(10)
wine['quality'].value_counts()
plt.show()


X=wine.drop('quality',axis=1)   #features minus what we are looking for 
y=wine['quality']               #features we are looking for
#training of data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)   #compare the data to actual data and then transform it
X_test=sc.fit_transform(X_test)
X_train[:10]
#Random forest model
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
pred_rfc[:20]
print(classification_report(y_test,pred_rfc))#how this model performed
print(confusion_matrix(y_test,pred_rfc))

#SVM model
svm=svm.SVC()
svm.fit(X_train,y_train)
pred_svm=svm.predict(X_test)
print(classification_report(y_test,pred_svm))
print(confusion_matrix(y_test,pred_svm))

 
