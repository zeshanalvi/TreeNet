from tensorflow.python.platform import tf_logging as logging
from keras.applications.densenet import DenseNet169
from keras.applications.resnet import ResNet152, ResNet50

from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
#from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input, merge, ZeroPadding2D, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import Sequential, backend as K, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adagrad


class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                'Reduce LR on plateau conditioned on metric `%s` which is not available. Available metrics are: %s',
                self.monitor, ','.join(list(logs.keys()))
            )

        if not self.monitor_op(current, self.best):  # not a new best
            if not self.in_cooldown():               # and we're not in cooldown
                if self.wait + 1 >= self.patience:   # going to reduce lr
                    print("Backtracking to best model before reducing LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs)  # actually reduce LR


class TreeNet:
  def __init__(self,selected_model_name,output_layer,weights,dataset,epochs,test_size,trained_weights_path):
    self.densenet169=DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    self.resnet152=ResNet152(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    #self.selected_model="dense"
    #self.output_layer='avg_pool'
    self.selected_model_name=selected_model_name
    self.output_layer=output_layer
    self.num_classes=dataset.num_classes
    self.epochs=epochs#5
    self.test_size=test_size#0.33
    self.init_weights=weights#'imagenet'
    self.trained_weights_path=trained_weights_path#"abc.h5"
    self.dataset=dataset

  
  def alter_last_layer(self):
    base_model=self.densenet169
    if(self.selected_model_name=="resnet"):
      base_model=self.resnet152
    x = base_model.get_layer(self.output_layer).output
    x = Dense(self.num_classes, name="output")(x)
    model = Model(base_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adag=Adagrad(learning_rate=0.01,initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
    model.compile(loss='mean_squared_error', optimizer=adag)
    return model
    
  def finetune(self):
    X_train, X_test, Y_train, Y_test = train_test_split(self.dataset.images, self.dataset.labels, test_size=self.test_size, random_state=5)
    self.densenet169=self.alter_last_layer()
    #callback=ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
    model_checkpoint_path = "/kaggle/working/"+self.selected_model_name+"_"+str(self.epoch)+"_updatingLR_best.h5"
    c1 = ModelCheckpoint(model_checkpoint_path,save_best_only=True,monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
    c2 = ReduceLRBacktrack(best_path=model_checkpoint_path, monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
    c3 = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.000001)
    self.densenet169.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=0,validation_data=(X_test, Y_test),callbacks=[c1,c2])
    self.densenet169.save(self.trained_weights_path)
    return self.densenet169

def prediction(self,finetuned_model,chunk_size=500):
    paths,Y_true1,Y_true=self.dataset.gather_paths_all()
    Y_pred=np.zeros((len(Y_true1)),float)
    Y_preds=np.zeros((len(Y_true1),self.dataset.num_classes),float)
    
    data_size=len(paths)
    chunks=int(data_size/chunk_size)
    for chunk in range(chunks+1):
        st,end=compute_chunk(data_size,chunk,chunk_size,chunks)
        if(st==end):
          break
        print("Progress of dataset ",data_name," is at ",st, " to ",end,)
        X=get_images(paths[st:end])
        Y=finetuned_model.predict(X)
        for i in range(st,end):
          Y_pred[i]=np.argmax(Y[i-st])+1
          Y_preds[i,:]=Y[i-st,:]
    return Y_true,Y_pred,Y_preds

def CombinerNet(num_classes=16):
    model = Sequential()
    model.add(layers.Dense(32, activation="relu", name="L1"))
    model.add(layers.Dense(num_classes, activation="softmax", name="L2"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def compute_chunk(data_size,chunk,chunk_size,chunks):
  st=chunk*chunk_size
  end=(chunk+1)*chunk_size
  if(end>data_size):
    end=data_size
  return st,end
