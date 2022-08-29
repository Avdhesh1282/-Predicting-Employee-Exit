# -Predicting-Employee-Exit
The data consists of categorical &amp; number data. Thus, needs data preprocessing. Make use of preprocessing techniques that you have learnt &amp; build model to predict 'left' named column
import pandas as pd

emp_data = pd.read_csv('Data/HR_comma_sep.csv.txt')

emp_data.head()

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	sales 	salary
0 	0.38 	0.53 	2 	157 	3 	0 	1 	0 	sales 	low
1 	0.80 	0.86 	5 	262 	6 	0 	1 	0 	sales 	medium
2 	0.11 	0.88 	7 	272 	4 	0 	1 	0 	sales 	medium
3 	0.72 	0.87 	5 	223 	5 	0 	1 	0 	sales 	low
4 	0.37 	0.52 	2 	159 	3 	0 	1 	0 	sales 	low

emp_data.rename(columns={'sales':'dept'}, inplace=True)

emp_data.head()

 	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years 	dept 	salary
0 	0.38 	0.53 	2 	157 	3 	0 	1 	0 	sales 	low
1 	0.80 	0.86 	5 	262 	6 	0 	1 	0 	sales 	medium
2 	0.11 	0.88 	7 	272 	4 	0 	1 	0 	sales 	medium
3 	0.72 	0.87 	5 	223 	5 	0 	1 	0 	sales 	low
4 	0.37 	0.52 	2 	159 	3 	0 	1 	0 	sales 	low

import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline

emp_data.describe()

	satisfaction_level 	last_evaluation 	number_project 	average_montly_hours 	time_spend_company 	Work_accident 	left 	promotion_last_5years
count 	14999.000000 	14999.000000 	14999.000000 	14999.000000 	14999.000000 	14999.000000 	14999.000000 	14999.000000
mean 	0.612834 	0.716102 	3.803054 	201.050337 	3.498233 	0.144610 	0.238083 	0.021268
std 	0.248631 	0.171169 	1.232592 	49.943099 	1.460136 	0.351719 	0.425924 	0.144281
min 	0.090000 	0.360000 	2.000000 	96.000000 	2.000000 	0.000000 	0.000000 	0.000000
25% 	0.440000 	0.560000 	3.000000 	156.000000 	3.000000 	0.000000 	0.000000 	0.000000
50% 	0.640000 	0.720000 	4.000000 	200.000000 	3.000000 	0.000000 	0.000000 	0.000000
75% 	0.820000 	0.870000 	5.000000 	245.000000 	4.000000 	0.000000 	0.000000 	0.000000
max 	1.000000 	1.000000 	7.000000 	310.000000 	10.000000 	1.000000 	1.000000 	1.000000
Preprocessing

emp_data.select_dtypes('object').columns

Index(['dept', 'salary'], dtype='object')

emp_data.dept.value_counts()

sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: dept, dtype: int64

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

dept = le.fit_transform(emp_data.dept)

ohe = OneHotEncoder()

ohe_dept = ohe.fit_transform(dept.reshape(-1,1))

ohe.active_features_

array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64)

le.classes_

array(['IT', 'RandD', 'accounting', 'hr', 'management', 'marketing',
       'product_mng', 'sales', 'support', 'technical'], dtype=object)

dept_df = pd.DataFrame(ohe_dept.toarray(), dtype=int,columns=le.classes_)

emp_data['salary_tf'] = emp_data.salary.map({'low':1,'medium':2,'high':3})

from sklearn.preprocessing import StandardScaler,MinMaxScaler

emp_data.columns

Index(['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years', 'dept', 'salary', 'salary_tf'],
      dtype='object')

df = emp_data[['number_project','average_montly_hours', 'time_spend_company']]

df.plot.kde()

<matplotlib.axes._subplots.AxesSubplot at 0x159e3514ac8>

mm = MinMaxScaler()

scaled_np = mm.fit_transform(df)

dept_np = dept_df.values

emp_df = emp_data[['satisfaction_level','last_evaluation','Work_accident','promotion_last_5years','salary_tf']]

emp_np = emp_df.values

feature_data = np.hstack([emp_np, scaled_np, dept_np])

target_data = emp_data.left

feature_data.shape

(14999, 18)

target_data.value_counts()

0    11428
1     3571
Name: left, dtype: int64

Model Building

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier

models = [ LogisticRegression(class_weight='balanced'), SGDClassifier(max_iter=10), PassiveAggressiveClassifier(max_iter=20), RandomForestClassifier(n_estimators=20)]

from sklearn.model_selection import train_test_split

trainX,testX,trainY,testY = train_test_split(feature_data,target_data)

for model in models:
    model.fit(trainX,trainY)
    print (model.score(testX,testY))

0.7626666666666667
0.8133333333333334
0.6269333333333333
0.9861333333333333

