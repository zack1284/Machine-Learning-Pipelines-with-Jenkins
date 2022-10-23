import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow
import os 
# Use IP of your remote machine here
server_ip = "172.23.120.50"

# set up minio credentials and connection
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'adminadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{server_ip}:9000"

# set mlflow track uri
mlflow.set_tracking_uri(f"http://{server_ip}:5000")
mlflow.set_experiment("Credit Card ML")

data=["gongguan_best.csv"]
for i in data:
    with mlflow.start_run(run_name=i):
        bike=pd.read_csv(i)
        x=bike.drop(["lent"],axis=1) #自變數
        y=bike["lent"]               #應變數

        train_x, valid_x, train_y, valid_y = train_test_split(x,y,test_size=0.3)

        lm=LinearRegression()
        lm.fit(train_x,train_y) #用訓練資料建構回歸模型
        rsquare=lm.score(train_x,train_y)

        predicted_y=lm.predict(valid_x) #代入驗證資料集的自變數，求得預測值
        rss=((predicted_y - valid_y)**2).mean()
        tss=((valid_y.mean()-valid_y)**2).mean()
        verror=1-(rss/tss)

        mlflow.log_param("num",len(x.columns))
        mlflow.log_metric("rsquare",rsquare) #訓練誤差
        mlflow.log_metric("verror",verror) #驗證誤差
        mlflow.sklearn.log_model(lm,"model")