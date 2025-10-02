from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
import numpy as np, pandas as pd, random
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import torch.nn as nn

class TreeNet(nn.Module):
  def __init__(self,layer_count=2,breath_count=1,classifier=True):
    super(TreeNet, self).__init__()
    self.layers={}
    self.classifier=classifier
    self.layer_count=layer_count
    self.breath_count=breath_count
    for i in range(layer_count):
      self.layers["layer_"+str(i)]={}
      if(i%2==0):
        for b in range(self.breath_count):
          if(classifier):
            self.layers["layer_"+str(i)]["RF_1_"+str(b)]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["RF_2_"+str(b)]=RandomForestClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["ET_3_"+str(b)]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["ET_4_"+str(b)]=ExtraTreesClassifier(n_estimators=random.randint(1,8)*25, random_state=42)
          else:
            self.layers["layer_"+str(i)]["RF_1_"+str(b)]=RandomForestRegressor(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["RF_2_"+str(b)]=RandomForestRegressor(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["ET_3_"+str(b)]=ExtraTreesRegressor(n_estimators=random.randint(1,8)*25, random_state=42)
            self.layers["layer_"+str(i)]["ET_4_"+str(b)]=ExtraTreesRegressor(n_estimators=random.randint(1,8)*25, random_state=42)
      else:
        for b in range(self.breath_count):
          if(classifier):
            self.layers["layer_"+str(i)]["XG_1_"+str(b)]=XGBClassifier()
            self.layers["layer_"+str(i)]["CB_2_"+str(b)]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
            self.layers["layer_"+str(i)]["CB_3_"+str(b)]=CatBoostClassifier(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='MultiClass', random_state=42)
            self.layers["layer_"+str(i)]["BC_4_"+str(b)]=BaggingClassifier(DecisionTreeClassifier(max_depth=10, random_state=42), n_estimators=random.randint(1,8)*25, random_state=42)
          else:
            self.layers["layer_"+str(i)]["XG_1_"+str(b)]=XGBRegressor()
            self.layers["layer_"+str(i)]["CB_2_"+str(b)]=CatBoostRegressor(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='RMSE', random_state=42)
            self.layers["layer_"+str(i)]["CB_3_"+str(b)]=CatBoostRegressor(iterations=random.randint(1,8)*25, learning_rate=0.1, depth=6, loss_function='RMSE', random_state=42)
            self.layers["layer_"+str(i)]["BC_4_"+str(b)]=BaggingRegressor(DecisionTreeRegressor(max_depth=10, random_state=42), n_estimators=random.randint(1,8)*25, random_state=42)
  
  def _ensure_dataframe(self, X):
    """Convert input (DataFrame or ndarray) into a pandas DataFrame"""
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        # create default column names col_0, col_1, ...
        cols = [f"col_{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["col_0"]
        return pd.DataFrame(X, columns=cols)
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray")
  
  def _ensure_numpy(self, X):
    """Convert input (DataFrame or ndarray) into a NumPy array"""
    if isinstance(X, pd.DataFrame):
        return X.values   # extract numpy array
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray")
          
  def train(self,trainX,trainY):
    trainX=self._ensure_dataframe(trainX)
    trainY=self._ensure_numpy(trainY)
    for l,layer in enumerate(self.layers):
      #print("Training Layer\t",l+1,"\t with input\t",trainX.shape)
      preds={}
      for i,forest in enumerate(self.layers[layer]):
        if("CB_" in forest):
          #trainX = trainX.apply(pd.to_numeric, downcast="float")
          if(self.classifier):
            preds[forest]=self.layers[layer][forest].fit(trainX, trainY.ravel(),verbose=0).predict_proba(trainX)
          else:
            preds[forest]=self.layers[layer][forest].fit(trainX, trainY.ravel(),verbose=0).predict(trainX)
        else:
          if(self.classifier):
            preds[forest]=self.layers[layer][forest].fit(trainX, trainY.ravel()).predict_proba(trainX)
          else:
            preds[forest]=self.layers[layer][forest].fit(trainX, trainY.ravel()).predict(trainX)
      for forest in self.layers[layer]:
        trainX = trainX.reset_index(drop=False)  # keep original string index in a column
        preds_df = pd.DataFrame(preds[forest]).add_suffix("_" + layer + "_" + forest)
        # concat by column (since order matches)
        trainX = pd.concat([trainX, preds_df], axis=1)
        # set string index back
        trainX = trainX.set_index(trainX.columns[0])
  def predict_prob(self,testX):
    testX=self._ensure_dataframe(testX)
    for l,layer in enumerate(self.layers):
      #print("Input Shape Before Layer ",layer,"\t",testX.shape)
      preds={}
      for i,forest in enumerate(self.layers[layer]):
        if(self.classifier):
          preds[forest]=self.layers[layer][forest].predict_proba(testX)
        else:
          preds[forest]=self.layers[layer][forest].predict(testX)
      for forest in self.layers[layer]:
        testX = testX.reset_index(drop=False)  # keep original string index in a column
        preds_df = pd.DataFrame(preds[forest]).add_suffix("_" + layer + "_" + forest)
        # concat by column (since order matches)
        testX = pd.concat([testX, preds_df], axis=1)
        # set string index back
        testX = testX.set_index(testX.columns[0])
    return np.mean(np.stack(list(preds.values())), axis=0)

    
  def predict(self,testX):
    testX=self._ensure_dataframe(testX)
    for l,layer in enumerate(self.layers):
      #print("Input Shape Before Layer ",layer,"\t",testX.shape)
      preds={}
      for i,forest in enumerate(self.layers[layer]):
        if(self.classifier):
          preds[forest]=self.layers[layer][forest].predict_proba(testX)
        else:
          preds[forest]=self.layers[layer][forest].predict(testX)
      for forest in self.layers[layer]:
        testX = testX.reset_index(drop=False)  # keep original string index in a column
        preds_df = pd.DataFrame(preds[forest]).add_suffix("_" + layer + "_" + forest)
        # concat by column (since order matches)
        testX = pd.concat([testX, preds_df], axis=1)
        # set string index back
        testX = testX.set_index(testX.columns[0])
        #testX=testX.merge(pd.DataFrame(preds[forest]).add_suffix("_"+layer+"_"+forest), left_index=True, right_index=True,copy=True)
    #print(preds.values())
    if(self.classifier):
      pred1 = np.argmax(np.mean(np.stack(list(preds.values())), axis=0), axis=1).reshape(-1, 1)
    else:
      pred1 = np.mean(np.stack(list(preds.values())), axis=0).reshape(-1, 1)
    return pred1



    
  def summary(self):
    print("Model has "+str(self.layer_count)," layers")
    print("Model Layer Breath is "+str(self.breath_count))
    print("Model Details")
    print(self.layers)

