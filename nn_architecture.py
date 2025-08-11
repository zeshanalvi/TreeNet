import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score, matthews_corrcoef, precision_score, recall_score
#classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
#!pip install xgboost
#!pip install lightgbm
#!pip install catboost

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import random


base_path="/kaggle/input/kvasirv1/"
class_count=8
chunk=0.5
files=[base_path+f for f in os.listdir(base_path) if ".csv" in f and "dev_" in f and "_500_" not in f and "Haralick" not in f and "lbp." not in f]# and "deep_" not in f]
files.sort()
print(len(files),files)
for f in files:
    df=pd.read_csv(f)
    #print(df.columns)
    df['filename']=df['class1']+"_"+df['img']
    
    df=df.sort_values(by='filename')
    print(f.split("/")[-1],df.shape,df['filename'][:5])

class TreeNet:
  def __init__(self,selected_model_name,output_layer,weights,dataset,epochs,test_size,trained_weights_path,batch_size=16):
    self.layers={}
    self.layer_count=2
    
    #self.densenet169=DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    #self.resnet152=ResNet152(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    #self.selected_model="dense"
    #self.output_layer='avg_pool'
    #self.selected_model_name=selected_model_name
    #self.output_layer=output_layer
    self.num_classes=dataset.num_classes
    self.epochs=epochs#5
    self.test_size=test_size#0.33
    self.init_weights=weights#'imagenet'
    self.trained_weights_path=trained_weights_path#"abc.h5"
    self.dataset=dataset
    self.batch_size=batch_size

  def read_all(files_dev):
    train_X=pd.DataFrame()
    train_Y=pd.DataFrame()

    for f in files_dev:
        print(f,train_X.shape)
        if("lbp" in f):
            df=pd.read_csv(f).iloc[:,1:]
            df["file_name"]=df['Yl']+"__"+df['filename'].astype('str')+".jpg"
            df.set_index("file_name",inplace=True)
            train_Y=df['Yl']
            df.drop(columns=['Yl','Y','filename','Unnamed: 0'],axis=1,inplace=True,errors='ignore')
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X=train_X.merge(df,on='key',how='inner',suffixes=("","_"+file_name.split('/')[-1].split(".")[0][4:9]))

        elif("densenet" in f):
            df=pd.read_csv(f).iloc[:,1:]
            df[['Y','img']] = df['image_name'].str.split('__',expand=True)
            df.rename(columns={"image_name": "file_name"},inplace=True)
            df.set_index("file_name",inplace=True)
            
            train_Y=df['Y']
            df.drop(['Actual','Pred','Y','img','Unnamed: 0'],axis=1,inplace=True,errors='ignore')
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X=train_X.merge(df,on='key',how='inner',suffixes=("","_"+file_name.split('/')[-1].split(".")[0][4:9]))
        elif("mobilenetv2" in f):
            df=pd.read_csv(f).iloc[:,1:-2].set_index('image_name')
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X=train_X.merge(df,on='key',how='inner',suffixes=("","_"+file_name.split('/')[-1].split(".")[0][4:9]))
        else:
            df=pd.read_csv(f).iloc[:,1:]
            df["file_name"]=df['class1']+"__"+df['img'].astype('str')
            df.set_index("file_name",inplace=True)
            train_Y=df['class1']
            df.drop(['class1','img','Unnamed: 0'],axis=1,inplace=True,errors='ignore')
            print(train_X.shape,df.shape)
            if(train_X.shape[0]==0):
                train_X=df
            else:
                train_X=train_X.merge(df,on='key',how='inner',suffixes=("","_"+file_name.split('/')[-1].split(".")[0][4:9]))

    print(train_X.shape,train_Y.shape)
    return train_X,train_Y

  def design_network(self):
    for i in range(self.layer_count):
      self.layers["layer_"+str(i)]={}
      if(i%2==0):
        self.layers["layer_"+str(i)]["RF_1"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["RF_2"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["RF_3"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["RF_4"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["RF_5"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        self.layers["layer_"+str(i)]["RF_6"]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        self.layers["layer_"+str(i)]["ET_7"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["ET_8"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["ET_9"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        self.layers["layer_"+str(i)]["ET_10"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["ET_11"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
        #self.layers["layer_"+str(i)]["ET_12"]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
      else:
        #self.layers["layer_"+str(i)]["XG_1"]=XGBClassifier()
        self.layers["layer_"+str(i)]["XG_2"]=XGBClassifier()
        #self.layers["layer_"+str(i)]["CB_3"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        self.layers["layer_"+str(i)]["CB_4"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        #self.layers["layer_"+str(i)]["CB_5"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        self.layers["layer_"+str(i)]["CB_6"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        #self.layers["layer_"+str(i)]["CB_7"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        #self.layers["layer_"+str(i)]["CB_8"]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
        #self.layers["layer_"+str(i)]["BC_9"]=BaggingClassifier(DecisionTreeClassifier(), n_estimators=random.randint(1,4)*50, random_state=42)
        #self.layers["layer_"+str(i)]["BC_10"]=BaggingClassifier(DecisionTreeClassifier(), n_estimators=random.randint(1,4)*50, random_state=42)
        self.layers["layer_"+str(i)]["BC_11"]=BaggingClassifier(DecisionTreeClassifier(), n_estimators=random.randint(1,4)*50, random_state=42)
        #self.layers["layer_"+str(i)]["BC_12"]=BaggingClassifier(DecisionTreeClassifier(), n_estimators=random.randint(1,4)*50, random_state=42)


train_Y, uniques = pd.factorize(train_Y)
print(train_Y)  # mapped values from 0 to 7 (or 0 to n-1)
print(uniques)   # original values corresponding to each label



  def portion(self,chunk=0.5):
    trainX=train_X[0] #merge all
    for i,X in enumerate(train_X[1:]):
      trainX=trainX.merge(X,left_index=True,right_index=True,how='inner',suffixes=("","_"+files[i+1].split('/')[-1].split(".")[0][4:9]))
    #train_Y-=1
    shuffled_indices = np.random.permutation(len(trainX))
    # Apply the same shuffle to both
    trainX = trainX.iloc[shuffled_indices].reset_index(drop=True)
    train_Y = train_Y[shuffled_indices]
    # Select 50% of the rows
    half_size = int(len(trainX)*chunk)
    trainX = trainX.iloc[:half_size]
    train_Y = train_Y[:half_size]
    return trainX,train_Y

  def fit_model(self):
    for l,layer in enumerate(layers):
      print("Input Shape Before Layer ",layer,"\t",trainX.shape)
      preds={}
      for i,forest in enumerate(layers[layer]):
        if(l==0):
          print("Started\n Forest\t",forest,"\t of Layer:",l+1)
          preds[forest]=layers[layer][forest].fit(trainX, train_Y).predict_proba(trainX)
        else:
          print("Started\n Forest\t",forest,"\t of Layer:",l+1)
          if("CB_" in forest):
            #print("CB_ started")
            preds[forest]=layers[layer][forest].fit(trainX, train_Y,verbose=0).predict_proba(trainX)
          else:
            preds[forest]=layers[layer][forest].fit(trainX, train_Y).predict_proba(trainX)
      for forest in layers[layer]:
        print(trainX.shape,)
        trainX=trainX.merge(pd.DataFrame(preds[forest]).add_suffix("_"+layer+"_"+forest), left_index=True, right_index=True,copy=True)
      print(len(preds))



train_X=[]
test_X=[]
train_Y=None
test_Y=None
for f in files:
    print(f.split("/")[-1])
    df=pd.read_csv(f)
    df['filename']=df['class1']+"_"+df['img']
    #df['filename']=df['labels']+"_"+df['filename']
    df=df.sort_values(by='filename')
    train_X.append(df.drop(columns={'class','img','class1','filename'},axis=1))
    train_Y=df["class"]-1
    print(df.shape,len(train_X),train_X[0].shape)

for f in files:
    f=f.replace("dev_","val_")
    print(f)
    df=pd.read_csv(f)
    df['filename']=df['class1']+"_"+df['img']
    #df['filename']=df['labels']+"_"+df['filename']
    df=df.sort_values(by='filename')
    
    test_X.append(df.drop(columns={'class','img','class1','filename'},axis=1))
    #test_X.append(df.drop(columns={'labels','filename','Y'},axis=1))
    test_Y=df["class"]-1
    print(f.split("/")[-1],df.shape,len(test_X),test_X[0].shape)
df=None
#train_X=pd.read_csv(f).drop(columns={'labels','filename','Y'},axis=1)