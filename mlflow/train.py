import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import mlflow
import os 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


filepath1 = 'creditcard.csv'
data = pd.read_csv(filepath1)


# Use IP of your remote machine here
server_ip = "172.23.120.50"

# set up minio credentials and connection
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'adminadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{server_ip}:9000"

# set mlflow track uri
mlflow.set_tracking_uri(f"http://{server_ip}:5000")
mlflow.set_experiment("Credit Card ML")


#in data: define X,y
X = data.drop('Class',axis = 1)
y = data['Class']

# split data into train and test set
X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#undersampling
undersample = RandomUnderSampler(sampling_strategy = 0.05)
X_under, y_under = undersample.fit_resample(X_train, y_train)


with mlflow.start_run(run_name="Logistic_imbalanced:"):

    logistic_imbalanced = linear_model.LogisticRegression()
    logistic_imbalanced.fit(X_train,y_train)
    #accuracy evaluation
    logistic_predictions_imbalanced = logistic_imbalanced.predict(X_test)
    
    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_train))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, logistic_predictions_imbalanced))
    mlflow.log_metric("Recall" , recall_score(y_test, logistic_predictions_imbalanced))
    mlflow.log_metric("Precision", precision_score(y_test, logistic_predictions_imbalanced))
    mlflow.log_metric("F1", f1_score(y_test, logistic_predictions_imbalanced))
    mlflow.sklearn.log_model(logistic_imbalanced,"Logistic Regression Imbalanced")
    print("Classification report:\n", classification_report(y_test, logistic_predictions_imbalanced))
    print(confusion_matrix(y_test, logistic_predictions_imbalanced))

with mlflow.start_run(run_name="Logistic_balanced:"):

    logistic_balanced = linear_model.LogisticRegression()
    logistic_balanced.fit(X_under,y_under)
    logistic_balanced_predictions = logistic_balanced.predict(X_test)
    
    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_under))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, logistic_balanced_predictions))
    mlflow.log_metric("Recall" , recall_score(y_test, logistic_balanced_predictions))
    mlflow.log_metric("Precision", precision_score(y_test, logistic_balanced_predictions))
    mlflow.log_metric("F1", f1_score(y_test, logistic_balanced_predictions))
    mlflow.sklearn.log_model(logistic_balanced,"Logistic Regression Balanced")
    print("Classification report:\n", classification_report(y_test, logistic_balanced_predictions))
    print(confusion_matrix(y_test, logistic_balanced_predictions))

with mlflow.start_run(run_name="RFC_Imbalanced:"):

    randfor_imbalanced = RandomForestClassifier(n_estimators = 40)
    randfor_imbalanced.fit(X_train, y_train)
    rf_predict_imbalanced = randfor_imbalanced.predict(X_test)
    
    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_train))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, rf_predict_imbalanced))
    mlflow.log_metric("Recall" , recall_score(y_test, rf_predict_imbalanced))
    mlflow.log_metric("Precision", precision_score(y_test, rf_predict_imbalanced))
    mlflow.log_metric("F1", f1_score(y_test, rf_predict_imbalanced))
    mlflow.sklearn.log_model(randfor_imbalanced,"RFC Imbalanced")
    print("Classification report:\n", classification_report(y_test, rf_predict_imbalanced))
    print(confusion_matrix(y_test, rf_predict_imbalanced))
    
with mlflow.start_run(run_name="RFC_Balanced:"):

    randfor_balanced = RandomForestClassifier(n_estimators = 40)
    randfor_balanced.fit(X_under, y_under)
    rf_predict_balanced = randfor_balanced.predict(X_test)

    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_under))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, rf_predict_balanced))
    mlflow.log_metric("Recall" , recall_score(y_test, rf_predict_balanced))
    mlflow.log_metric("Precision", precision_score(y_test, rf_predict_balanced))
    mlflow.log_metric("F1", f1_score(y_test, rf_predict_balanced))
    mlflow.sklearn.log_model(randfor_balanced,"RFC Balanced")
    print("Classification report:\n", classification_report(y_test, rf_predict_balanced))
    print(confusion_matrix(y_test, rf_predict_balanced))

with mlflow.start_run(run_name="XGB_Imbalanced:"):

    xgbmodel_imbalanced = XGBClassifier(n_estimator = 100, learning_rate = 0.3)
    xgbmodel_imbalanced.fit(X_train, y_train)
    xgb_predict_imbalanced = xgbmodel_imbalanced.predict(X_test)
    
    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_train))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, xgb_predict_imbalanced))
    mlflow.log_metric("Recall" , recall_score(y_test, xgb_predict_imbalanced))
    mlflow.log_metric("Precision", precision_score(y_test, xgb_predict_imbalanced))
    mlflow.log_metric("F1", f1_score(y_test, xgb_predict_imbalanced))
    mlflow.sklearn.log_model(xgbmodel_imbalanced,"XGB Imbalanced")
    print("Classification report:\n", classification_report(y_test, xgb_predict_imbalanced))
    print(confusion_matrix(y_test, xgb_predict_imbalanced))


with mlflow.start_run(run_name="XGB_Balanced:"):

    xgbmodel_balanced = XGBClassifier(n_estimator = 100, learning_rate = 0.3)
    xgbmodel_balanced.fit(X_under, y_under)
    xgb_predict_balanced = xgbmodel_balanced.predict(X_test)
    
    #mlflow
    param=data.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_train))
    mlflow.log_param("Test rows", len(X_test))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, xgb_predict_balanced))
    mlflow.log_metric("Recall" , recall_score(y_test, xgb_predict_balanced))
    mlflow.log_metric("Precision", precision_score(y_test, xgb_predict_balanced))
    mlflow.log_metric("F1", f1_score(y_test, xgb_predict_balanced))
    mlflow.sklearn.log_model(xgbmodel_balanced,"XGB Balanced")
    #print result
    print("Classification report:\n", classification_report(y_test, xgb_predict_balanced))
    print(confusion_matrix(y_test, xgb_predict_balanced))







