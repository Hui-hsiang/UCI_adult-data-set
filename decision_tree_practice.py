import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing


col_names = ['age','workclass','fnlwht','education','education-num','marital-status'
				   ,'occupation','relationship','race','sex','capital-gain','capital-loss',
				   'hours-per-week','native-country','result' ]
data = pd.read_csv("adult.csv",names = col_names)

data_clean = data.replace(regex=[r'\?|\.|\$'],value=np.nan)

adult = data_clean.dropna(how='any')

label_encoder = preprocessing.LabelEncoder()
for col in col_names:
    if (col in ['fnlwht','education-num','capital-gain','capital-loss','hours-per-week','age'] ):
        continue
    encoded = label_encoder.fit_transform(adult[col])
    adult[col] = encoded

X_train , X_test , y_train , y_test = train_test_split(adult[col_names[:14]],adult[col_names[14]],test_size=0.3,random_state=1010)




sc=StandardScaler()

sc.fit(X_train)
x_train_nor=sc.transform(X_train)
x_test_nor=sc.transform(X_test)

tree=DecisionTreeClassifier(criterion='entropy') 
tree_clf=tree.fit(x_train_nor,y_train)

y_test_predicted = tree_clf.predict(x_test_nor)

accuracy = accuracy_score(y_test, y_test_predicted)
print('準確率:',accuracy)
