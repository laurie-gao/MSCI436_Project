import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from imblearn import over_sampling
from sklearn import metrics
import pickle

print("Loading data...")
fraud = pd.read_csv('data/fraudTrain.csv')
fraud_test = pd.read_csv('data/fraudTest.csv')

print("Preprocessing...")
# Drop columns with little significance to determining fraud 
fraud.drop(['cc_num', 'first', 'last', 'street', 'trans_num'], axis=1, inplace=True)
fraud.drop(fraud.iloc[:,[0]], axis=1, inplace=True)
fraud_test.drop(['cc_num', 'first', 'last', 'street', 'trans_num'], axis=1, inplace=True)
fraud_test.drop(fraud_test.iloc[:,[0]], axis=1, inplace=True)

# Converting date of birth (dob) to age
fraud['dob'] = pd.to_datetime(fraud['dob'])
fraud['age'] = (pd.to_datetime('now') - fraud['dob'])/ np.timedelta64(1, 'Y')
fraud['age'] = fraud['age'].astype(int)
fraud.drop(['dob'], axis=1, inplace=True)

fraud_test['dob'] = pd.to_datetime(fraud_test['dob'])
fraud_test['age'] = (pd.to_datetime('now') - fraud_test['dob'])/ np.timedelta64(1, 'Y')
fraud_test['age'] = fraud_test['age'].astype(int)
fraud_test.drop(['dob'], axis=1, inplace=True)

# Splitting trans_date_trans_time column into trans_date and trans_time
fraud['trans_date'] = pd.DatetimeIndex(fraud['trans_date_trans_time']).date
fraud['trans_time'] = pd.DatetimeIndex(fraud['trans_date_trans_time']).time
fraud.drop(['trans_date_trans_time'], axis=1, inplace=True)

fraud_test['trans_date'] = pd.DatetimeIndex(fraud_test['trans_date_trans_time']).date
fraud_test['trans_time'] = pd.DatetimeIndex(fraud_test['trans_date_trans_time']).time
fraud_test.drop(['trans_date_trans_time'], axis=1, inplace=True)

# Transform "merchant" into numeric variable
label_encoder = LabelEncoder()
fraud.merchant = label_encoder.fit_transform(fraud.merchant)
fraud_test.merchant = label_encoder.fit_transform(fraud_test.merchant)

# Transform "city" into numeric variable
fraud.city = label_encoder.fit_transform(fraud.city)
fraud_test.city = label_encoder.fit_transform(fraud_test.city)

# Transform "category" into numeric variable
fraud.category = label_encoder.fit_transform(fraud.category)
fraud_test.category = label_encoder.fit_transform(fraud_test.category)

# Transform "gender" into numeric variable
fraud.gender = fraud.gender.map({'M': 1, "F": 0})
fraud_test.gender = fraud_test.gender.map({'M': 1, "F": 0})

# Transform "state" into numeric variable
fraud.state = label_encoder.fit_transform(fraud.state)
fraud_test.state = label_encoder.fit_transform(fraud_test.state)

# Transform "job" into numeric variable
fraud.job = label_encoder.fit_transform(fraud.job)
fraud_test.job = label_encoder.fit_transform(fraud_test.job)

# Convert trans_time into seconds
fraud['trans_date'] =  pd.to_datetime(fraud['trans_date'])
fraud.trans_date = fraud.trans_date.map(dt.datetime.toordinal)
fraud.trans_time = pd.to_datetime(fraud.trans_time,format='%H:%M:%S')
fraud.trans_time = 3600 * pd.DatetimeIndex(fraud.trans_time).hour + 60 * pd.DatetimeIndex(fraud.trans_time).minute + pd.DatetimeIndex(fraud.trans_time).second

fraud_test['trans_date'] =  pd.to_datetime(fraud_test['trans_date'])
fraud_test.trans_date = fraud_test.trans_date.map(dt.datetime.toordinal)
fraud_test.trans_time = pd.to_datetime(fraud_test.trans_time,format='%H:%M:%S')
fraud_test.trans_time = 3600 * pd.DatetimeIndex(fraud_test.trans_time).hour + 60 * pd.DatetimeIndex(fraud_test.trans_time).minute + pd.DatetimeIndex(fraud_test.trans_time).second

# Seperate target from variables
X_train = fraud.drop('is_fraud', axis=1)
y_train = fraud['is_fraud']

X_test = fraud_test.drop('is_fraud', axis=1)
y_test = fraud_test['is_fraud']

# Variables to be scaled
vars_to_scale = ['merchant', 'category', 'amt', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long', 'age', 'trans_date', 'trans_time']

# Scale the variables
# scaler = QuantileTransformer(output_distribution='normal')
# scaler.fit(X_train[vars_to_scale])
# print(X_test[vars_to_scale].describe())
# X_train[vars_to_scale] = scaler.transform(X_train[vars_to_scale])
# X_test[vars_to_scale] = scaler.transform(X_test[vars_to_scale])

# Address imbalance using over sampling
ro = over_sampling.RandomOverSampler(random_state=100)
X_train_ro, y_train_ro = ro.fit_resample(X_train, y_train)

print("Fitting Model...")
# Fit the model
xgb = XGBClassifier(learning_rate=0.5, max_depth=10, n_estimators=15, max_features = 14)
xgb.fit(X_train_ro, y_train_ro)
y_test_pred = xgb.predict(X_test)
print ('AUC         : ', metrics.roc_auc_score(y_test, y_test_pred))
unique, counts = np.unique(y_test_pred, return_counts=True)
print(unique, counts)

print("Saving model and scaler...")
pickle.dump(xgb, open("model.pkl", "wb"))
# pickle.dump(scaler, open("scaler.pkl", "wb"))

model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))
# print(X_test[vars_to_scale].describe())
# X_test[vars_to_scale] = scaler.transform(X_test[vars_to_scale])
# print(X_test[vars_to_scale].describe())

y_pred = model.predict(X_test)
unique, counts = np.unique(y_pred, return_counts=True)
print(unique, counts)


print("Model Saved!")