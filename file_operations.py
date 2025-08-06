from sklearn.metrics import accuracy_score, f1_score

def gather_paths_all(jpg_path,num_classes=16):
  i=0
  if(num_classes==16):
    label_map=label_map16
  elif(num_classes==8):
    label_map=label_map8
  elif(num_classes==36):
    label_map=label_map36
  elif(num_classes==23):
    label_map=label_map23
  
  
  folder=os.listdir(jpg_path)
  count=0
  if (os.path.isfile(jpg_path+folder[0])):
    count=len(os.listdir(jpg_path))
  else:
    count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
  ima=['' for x in range(count)]
  labels=np.zeros((count,num_classes),dtype=float)
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
      for f in os.listdir(jpg_path+fldr+"/"):
          im=jpg_path+fldr+"/"+f
          ima[i]=im
          label[i]=label_map.index(fldr)+1
          i+=1
      if(count<=i):
          break
  for i in range(count):
      labels[i][label[i]-1]=1
  return ima,label,labels

label_map8=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']

def gather_paths_some(jpg_path,num_classes=16,fraction=0.5,label_map=label_map8):
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
    return ima,label,labels

def gather_images_from_paths(jpg_path,start,count,img_rows=224,img_cols=224):
    print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
    ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
    for i in range(count):
        #print(i,count,jpg_path[start+i])
        img=cv2.imread(jpg_path[start+i])
        im = cv2.resize(img, (img_rows, img_cols)).astype(np.uint8)
        ima[i]=im
    return ima

def get_images(paths,img_rows=224,img_cols=224):
    count=len(paths)
    ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
    for i in range(count):
        img=cv2.imread(paths[i])
        im = cv2.resize(img, (img_rows, img_cols)).astype(np.uint8)
        ima[i]=im
    return ima

def gather_images_from_paths_HSV(jpg_path,start,count,img_rows=224,img_cols=224):
  print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_path))
  ima=np.zeros((count,img_rows,img_cols,3),np.uint8)
  for i in range(count):
    img=cv2.imread(jpg_path[start+i])
    img_rgb = cv2.resize(img, (img_rows, img_cols)).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ima[i]=img_hsv
  return ima

def print_scores(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average='weighted')
  print(f"Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
  return acc
