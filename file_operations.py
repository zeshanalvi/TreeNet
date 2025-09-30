import os, pandas as pd
def subset(trainX,trainY,chunk=1):
  shuffled_indices = np.random.permutation(len(trainX))
  # Apply the same shuffle to both
  trainX = trainX.iloc[shuffled_indices].reset_index(drop=True)
  trainY = trainY[shuffled_indices]
  # Select 50% of the rows
  half_size = int(len(trainX)*chunk)
  trainX = trainX.iloc[:half_size]
  trainY = trainY[:half_size]

def read_files(base_path):
  #base_path="/kaggle/input/kvasirv1/"
  files=[base_path+f for f in os.listdir(base_path) if ".csv" in f and "dev_" in f and "_500_" not in f and "Haralick" not in f and "lbp." not in f]# and "deep_" not in f]
  files.sort()
  #print(len(files),files)
  trainX=None
  testX=None
  trainY=None
  testY=None
  for f in files:
    #print(f.split("/")[-1])
    df=pd.read_csv(f)
    df['filename']=df['class1']+"_"+df['img']
    #df['filename']=df['labels']+"_"+df['filename']
    df=df.sort_values(by='filename')
    if(trainX is None):
      trainX=df.drop(columns={'class','img','class1','filename'},axis=1)
      trainY=df["class"]-1
      trainY, _ = pd.factorize(trainY)
    else:
      trainX=trainX.merge(df.drop(columns={'class','img','class1','filename'},axis=1),left_index=True,right_index=True,how='inner',suffixes=("","_"+f.split('/')[-1].split(".")[0][4:9]))
      
    #print(df.shape,len(train_X),train_X[0].shape)

  for f in files:
    f=f.replace("dev_","val_")
    #print(f)
    df=pd.read_csv(f)
    df['filename']=df['class1']+"_"+df['img']
    #df['filename']=df['labels']+"_"+df['filename']
    df=df.sort_values(by='filename')
    if(testX is None):
      testX=df.drop(columns={'class','img','class1','filename'},axis=1)
      testY=df["class"]-1
      testY, _ = pd.factorize(testY)
    else:
      testX=testX.merge(df.drop(columns={'class','img','class1','filename'},axis=1),left_index=True,right_index=True,how='inner',suffixes=("","_"+f.split('/')[-1].split(".")[0][4:9]))
    #print(f.split("/")[-1],df.shape,len(test_X),test_X[0].shape)
  df=None
  return trainX,trainY,testX,testY


#read_files(base_path="/kaggle/input/kvasirv1/")