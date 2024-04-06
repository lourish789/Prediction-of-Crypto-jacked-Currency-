SEED = 1221
# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb

st.title("Cryptojacking Prediction Model")
st.image("crypto.png")
st.text("""
Cryptojacking is a cyber-attack utilizing malicious scripts similar to those from 
large cryptocurrency houses to illegally mine data without users being aware. 
These attacks are stealthy and difficult to detect or analyze, often leading to 
decreased computing speeds for users as well as crashes due to straining of
computational resources.


The goal of this project is to classify network activity from 
various websites as either cryptojacking or not 
based on features related to both network-based and host-based data.
""")

st.text("""Features of the data includes;
I/O Data Operations	
I/O Data Bytes
Number of subprocesses
Time on processor
Disk Reading/sec
Disc Writing/sec
Bytes Sent/sent
Received Bytes (HTTP)
Network packets sent
Network packets received
Pages Read/sec	
Pages Input/sec	
Page Errors/sec	
Confirmed byte radius
Label or Target
""")



train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
samplesubmission = pd.read_csv('SampleSubmission.csv')
st.dataframe(train.head())

fig = plt.figure(figsize=(12, 6))
sns.set_style('darkgrid')
sns.countplot(x = 'Label', data = train)
plt.title('Target Variable Distribution')
plt.show()
st.pyplot(fig)

#st.bar_chart(train["Label"].unique)
st.sidebar.header("Choose the Value for the features")
def choose_feature():
    feature = dict()
    IO1 = st.sidebar.slider("I/O Data Operations", -0.5, 0.5)
    IO2 = st.sidebar.slider("I/O Data Bytes", 0, 1000) 
    Subs = st.sidebar.slider("Number of subprocesses", 20, 50)
    Pros = st.sidebar.slider("Time on processor", 0.0, 1.0, 0.1)
    Page_read = st.sidebar.slider("Pages Read/seconds", 0, 30)
    Page_error = st.sidebar.slider("Page Errors/seconds", 100, 1000)
    Page_input = st.sidebar.slider("Pages Input/sec", 0, 10)
    confirm = st.sidebar.slider("Confirmed byte radius", 15, 35)
    HTTP = st.sidebar.slider("Received Bytes (HTTP)", 0, 1000)
    received = st.sidebar.slider("Network Packets Received", 0, 100)
    sent = st.sidebar.slider("Network packets sent", 0, 100)
    DWS = st.sidebar.slider("Disc Writing/sec", 0, 10)
    DRS = st.sidebar.slider("Disc Reading/sec", 0, 10)
    byte = st.sidebar.slider("Bytes Sent/sent", 0, 5000)
    feature["I/O Operations"] = IO1
    feature["I/O Data Bytes"] = IO2
    feature["Number of Subprocesses"] = Subs
    feature["Time on Processor"] = Pros
    feature["Pages Read/Seconds"] = Page_read
    feature["Page Errors/Seconds"] = Page_error
    feature["Pages Input/sec"] = Page_input
    feature["Confirmed Byte Radius"] = confirm
    feature["Received Bytes (HTTP)"] = HTTP
    feature["Network Packets Received"] = received
    feature["Network Packets Sent"] = sent
    feature["Disc Writing/Seconds"] = DWS
    feature["Disc Reading/Seconds"] = DRS
    feature["Bytes Sent/sent"] = byte
    cf = pd.DataFrame(feature, index=[0])
    return feature


feat = choose_feature()
st.subheader("User Input Parameters")
feat = dict(feat)
#feat.reshape(-1, 1)

#scale numerical features for logistic model
from sklearn.preprocessing import StandardScaler
features = train.drop(columns=['ID','Label']).columns
target = 'Label'

# define standard scaler
scaler = StandardScaler()

# transform data
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

#split train data into train and validation set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train[features], 
                                                 #   train[target].to_frame(),
                                                  #  stratify=train[target], #to account for class imbalance
                                                   # test_size=0.25,
                                                    #random_state=SEED)

import imblearn 
#from imblearn.over_sampling import SMOTE 
#smote = SMOTE(random_state=1) 
#X_train, y_train = smote.fit_resample(X_train, y_train)
X = train.drop(["ID", "Label"], axis=1)
y = train.Label
#sfeat = array.reshape(-1, 1)
st.subheader("Class Labels and their Coreresponding index numbers")
st.write(train.Label)

import lightgbm as ltb
ltb_model = ltb.LGBMClassifier(random_state=10)
ltb_model.fit(X, y)
pred = ltb_model.predict(feat)
pred_proba = ltb_model.predict_proba(feat)

#ltb_model.fit(X_train, y_train["Label"])
#ltb_y_pred = ltb_model.predict(feat)
print(ltb_y_pred)
#ltb_model.score(X_test, y_test)
#prediction_proba = ltb_model.prdict_proba(feat)


