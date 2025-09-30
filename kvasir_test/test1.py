import numpy as np, pandas as pd, string, random
from treenet import TreeNet as tn
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
    return trainX, trainY

def read_files(files):
    merged=None
    for i,f in enumerate(dev_files):
        df=pd.read_csv(f)
        df.set_index("img", inplace=True)
        Y=pd.DataFrame(df["class1"])
        df.drop(columns={'class','class1'},axis=1, inplace=True)
        if 'merged' not in locals():
            merged = df.copy(deep=True)
        else:
            merged = pd.concat([merged,df.add_suffix("_"+str(i))], axis=1, join="inner")
    return merged,Y


def testing_code():
    y=4
    n_samples=10
    col_labels = [''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(y)]

    X_train = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    X_test = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    y_train = np.random.randint(0, y, size=(n_samples,))

    print(type(X_train),type(X_test),type(y_train))

    model = tn(layer_count=3, breath_count=2)
    model.train(X_train, y_train)

    probs = model.predict_prob(X_test)
    print(probs.shape)

    preds = model.predict(X_test)
    print(preds.shape)



#testing_code()


print("start of code")

data_path="kvasirv1"
dev_files=[data_path+"/"+f for f in os.listdir(data_path) if "dev_" in f and ".csv" in f]
#print("Total Files Found",len(dev_files))

val_files=[data_path+"/"+f for f in os.listdir(data_path) if "val_" in f and ".csv" in f]
#print("Total Files Found",len(val_files))

train_X, train_y=read_files(dev_files)
test_X, test_y=read_files(val_files)

label_map8=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']

mapping = {cls: idx for idx, cls in enumerate(label_map8)}

# Replace values in Y['class'] with indices
train_y["class1"] = train_y["class1"].map(mapping)
test_y["class1"] = test_y["class1"].map(mapping)

#train_ys = pd.get_dummies(train_y["class1"])
#test_ys = pd.get_dummies(test_y["class1"])





train_ys = pd.DataFrame(np.eye(8)[train_y["class1"]],index=train_y.index)
test_ys = pd.DataFrame(np.eye(8)[test_y["class1"]],index=test_y.index)



trainy=train_y["class1"].to_numpy()
print(train_X.shape, trainy, test_X.shape, test_ys.shape)

print(train_X)
print(train_ys)



train_X,trainy=subset(train_X,trainy,chunk=0.1)


print(type(train_X),type(test_X),type(trainy))


model = tn(layer_count=3, breath_count=2)
print("model")
model.train(train_X, trainy)
print("trained")
probs = model.predict_prob(test_X)
print("test")
print(probs)


print("Acc Test:\t",accuracy_score(test_y, predY1))
print("Precision Test:\t",precision_score(test_y, predY1, average='weighted'))
print("Recall Test:\t",recall_score(test_y, predY1, average='weighted'))
print("F1-score Test:\t",f1_score(test_y, predY1, average='weighted'))
print("MCC Test:\t",matthews_corrcoef(test_y, predY1))

