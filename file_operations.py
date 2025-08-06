from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np, random, os, math, cv2, os, h5py, pandas as pd


class Dataset:
  def __init__(self, label_map,num_classes, images_base_path):
    self.label_map = label_map
    self.num_classes=num_classes
    self.images_base_path=images_base_path
    self.img_rows=224
    self.img_cols=224
    self.channels=3
    self.file_names=[]
    self.images=[]
    self.label=[]
    self.labels=[]

  def gather_paths_all(self):
    i=0
    jpg_path=self.images_base_path
    folder=os.listdir(jpg_path)
    count=0
    if (os.path.isfile(jpg_path+folder[0])): # Single Directory for all files
      count=len(os.listdir(jpg_path))
    else:
      count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)]) # Class wise directory
    ima=['' for x in range(count)] #Blank Image Path Array
    labels=np.zeros((count,num_classes),dtype=float) # Label Vectors
    label=[0 for x in range(count)] # Label Numbers
    if (os.path.isfile(jpg_path+folder[0])):
      for f in folder:
        im=jpg_path+f
        ima[i]=im
        label[i]=0
        i+=1
        if(count<i):
          break # All images processed
    else:
      for fldr in folder:
        for f in os.listdir(jpg_path+fldr+"/"):
            im=jpg_path+fldr+"/"+f
            ima[i]=im
            label[i]=label_map.index(fldr)+1
            i+=1
        if(count<=i):
            break # All images processed
    for i in range(count):
        labels[i][label[i]-1]=1 # Labels to Label Vectors
    return ima, label, labels

  def gather_paths_some(self,fraction=0.5):
    jpg_path=self.images_base_path
    print("Extracting Fraction of \t",fraction)
    i=0
    folder=os.listdir(jpg_path)
    count=0
    if (os.path.isfile(jpg_path+folder[0])):
        count=len(os.listdir(jpg_path))
    else:
        count=sum([int(len(os.listdir(jpg_path+f))) for f in os.listdir(jpg_path)])
    ima=['' for x in range(count)]
    label=[0 for x in range(count)]
    if (os.path.isfile(jpg_path+folder[0])):
        for f in folder:
            im=jpg_path+f
            ima[i]=im
            label[i]=0
            i+=1
            if(count<i):
                break
    else:
        for fldr in folder:
            br=0
            for f in os.listdir(jpg_path+fldr+"/"):
                im=jpg_path+fldr+"/"+f
                ima[i]=im
                label[i]=label_map.index(fldr)+1
                i+=1
                br+=1
                #print(br,i,math.ceil(len(os.listdir(jpg_path+fldr+"/"))/2))
                if(br>=math.ceil(len(os.listdir(jpg_path+fldr+"/")))):
                    break
                if(count<=i):
                    break
            if(count<=i):
                break
    
    #shuffle
    frac=int(fraction*len(ima))
    print("selected ",frac," images from ",len(ima))
    combined = list(zip(ima, label))
    random.shuffle(combined)
    selected = combined[:frac]
    ima_selected, label_selected = zip(*selected)
    ima = list(ima_selected)
    label = list(label_selected)    
    labels=np.zeros((frac,num_classes),dtype=float)    
    for i in range(frac):
        labels[i][label[i]-1]=1
    return ima, label, labels

  def gather_images_from_paths(self,start,count):
    jpg_path=self.images_base_path
    print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
    ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
    for i in range(count):
        #print(i,count,jpg_path[start+i])
        img=cv2.imread(jpg_path[start+i])
        im = cv2.resize(img, (self.img_rows, self.img_cols)).astype(np.uint8)
        ima[i]=im
    return ima

  def get_images(self):
    paths=self.file_names
    count=len(paths)
    ima=np.zeros((count,self.img_rows,self.img_cols,self.channels),np.uint8)
    for i in range(count):
        img=cv2.imread(paths[i])
        im = cv2.resize(img, (self.img_rows, self.img_cols)).astype(np.uint8)
        ima[i]=im
    return ima

  def gather_images_from_paths_HSV(self,start,count):
    jpg_path=self.images_base_path
    print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
    ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
    for i in range(count):
      img=cv2.imread(jpg_path[start+i])
      img_rgb = cv2.resize(img, (img_rows, img_cols)).astype(np.uint8)
      img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
      ima[i]=img_hsv
    return ima
