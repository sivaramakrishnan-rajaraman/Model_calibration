#declare gpu use
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #if using multiple gpus, otherwise comment

#%%
#clear warnings

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

#%%
#import other libraries

import time
import itertools
from itertools import cycle
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax
import numpy as np
from scipy import interp
from numpy import genfromtxt
import pandas as pd
import math
from tensorflow.keras.utils import to_categorical
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import seaborn as sns
import ml_insights as mli
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
print(mli.__version__)

#%% 
#get current working directory
os.getcwd()

#%% 
# define custom function for confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
#function to compute performance metrics

def perf_metrix(real_values,pred_values,beta):
   CM = confusion_matrix(real_values,pred_values)
   TP = CM[0][0]
   FN = CM[0][1] 
   FP = CM[1][0]
   TN = CM[1][1]
   Population = TN+FN+TP+FP
   Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN**2 + FN * TN + FP**2 + FP * TN)
   Prevalence = round( (TP+FP) / Population,2)
   Accuracy   = round( (TP+TN) / Population,4)
   Precision  = round( TP / (TP+FP),4 )
   NPV        = round( TN / (TN+FN),4 )
   FDR        = round( FP / (TP+FP),4 )
   FOR        = round( FN / (TN+FN),4 ) 
   check_Pos  = Precision + FDR
   check_Neg  = NPV + FOR
   Recall     = round( TP / (TP+FN),4 )
   FPR        = round( FP / (TN+FP),4 )
   FNR        = round( FN / (TP+FN),4 )
   TNR        = round( TN / (TN+FP),4 ) 
   check_Pos2 = Recall + FNR
   check_Neg2 = FPR + TNR
   LRPos      = round( Recall/FPR,4 ) 
   LRNeg      = round( FNR / TNR ,4 )
   DOR        = round( LRPos/LRNeg)
   F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
   FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
   MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
   BM         = Recall+TNR-1
   MK         = Precision+NPV-1
   mat_met = pd.DataFrame({
'Metric':['TP','TN','FP','FN','Kappa','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall','FPR','FNR','TNR','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK'],     'Value':[TP,TN,FP,FN,Kappa,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK]})

   return (mat_met)

#%%
#function to compute Expected calibration error (ECE)

def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

#%% Load data

img_width, img_height = 256,256
train_data_dir = "data/cardio_rsna_normal_cropped/train"
test_data_dir = "data/cardio_rsna_normal_cropped/test"
epochs = 32 
batch_size = 32 
num_classes = 2 #abnormal and normal
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators

datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)

#true labels
Y_test=validation_generator.classes
print(Y_test.shape)

#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes)
print(Y_test1.shape)

'''
Found 6962 images belonging to 2 classes.
Found 2983 images belonging to 2 classes.
{'abnormal': 0, 'normal': 1}
{'abnormal': 0, 'normal': 1}
(2983,)
(2983, 2)
'''
#%%
#declare model architecture

vgg16_cnn = VGG16(include_top=False, weights='imagenet', 
                        input_tensor=model_input)
vgg16_cnn.summary()

#%%
base_model_vgg16=Model(inputs=vgg16_cnn.input,
                        outputs=vgg16_cnn.get_layer('block5_conv3').output)
x = base_model_vgg16.output    
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, 
                    activation='softmax', name='predictions')(x)
model_vgg16 = Model(inputs=base_model_vgg16.input, 
                    outputs=predictions, 
                    name = 'vgg16_cardiomegaly')
model_vgg16.summary()

#%% train the vgg16 model

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  
model_vgg16.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 
filepath = 'weights/' + model_vgg16.name + '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='max', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_accuracy', 
                              patience=5, 
                              verbose=1, 
                              mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='max', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
model_vgg16_history = model_vgg16.fit(train_generator, 
                                          steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs, validation_data=validation_generator,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // batch_size, 
                                  verbose=1)

print('Training time: %s' % (time.time()-t))

#%% plot performance

N = 15 #epochs; change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_vgg16_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_vgg16_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
          model_vgg16_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_vgg16_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("VGG16_cardiomegaly_performance.png")

#%%
# Load model for evaluation

vgg16_model = load_model('working_weights_predictions/new_cardiomegaly/weights/vgg16_cardiomegaly.35-0.9415.h5')
vgg16_model.summary()

#compile the model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  
vgg16_model.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 

#%%

#Generator reset
validation_generator.reset()

#evaluate accuracy 
custom_y_pred = vgg16_model.predict(validation_generator,
                                    nb_validation_samples // batch_size, 
                                    verbose=1)

#%%

#we need the scores of only the positive abnormal class
custom_y_pred1 = custom_y_pred[:,1]

#%%
# plot roc curve and find the optimum threshold

#take only the abnormal class and plot
fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred[:,1])

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred[:,1])
print(auc_score)

# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))

# get the Youden's J statistic
J = tpr - fpr

# locate the index of the largest J statistic
ix_youden = argmax(J)

# locate the index of the largest g-means
ix_gmeans = argmax(gmeans)

#get the best threshold with G-means
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix_gmeans], gmeans[ix_gmeans]))

#get the best threshold with Youden's J statistic
print('Best Threshold=%f, Youden J statistic=%.3f' % (thresholds[ix_youden], J[ix_youden]))

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')

#select the optimum point using G-means
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='Model')

plt.scatter(fpr[ix_gmeans], 
               tpr[ix_gmeans], 
               marker='X',           
               s=300, color='blue', 
               label='Optimal threshold')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

'''
Best Threshold=0.840677, G-Mean=0.894
Best Threshold=0.840677, Youden J statistic=0.789
'''
#%%
'''
from Spline calibration that gave us the least expected calibration error
we obtained the best threshold for the model as shown below:

AUC: 0.9577718065316246
Best Threshold=0.887578, G-Mean=0.894
Best Threshold=0.887578, Youden J statistic=0.789

Using this threshold, we obtained the highest recall from the model.
We decided to experiment with a threshold greater than this value, 
say 0.9 to observe the performance before and after calibration

Hence, we use a threshold of 0.9 for the fourth table in the Results
Section
'''

custom_y_pred1_highsensitivity_opt = np.where(custom_y_pred1 > 0.9, 1, 0) 
print(custom_y_pred1_highsensitivity_opt)

#%%
#print all metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%
#using seaborn print confusion matrix
target_names = ['Abnormal','Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred1_highsensitivity_opt,
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred1_highsensitivity_opt)
np.set_printoptions(precision=5)

x_axis_labels = ['Cardiomegaly','No finding'] # labels for x-axis
y_axis_labels = ['Cardiomegaly','No finding'] # labels for y-axis

plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=True, cmap='Greens', #coolwarm
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
# Fit Platt scaling (logistic calibration)
lr = LogisticRegression(C=99999999999, solver='lbfgs')
lr.fit(custom_y_pred[:,1].reshape(-1,1), Y_test)
custom_y_pred1_platts1 = lr.predict_proba(custom_y_pred[:,1].reshape(-1,1))
custom_y_pred1_platts = lr.predict_proba(custom_y_pred[:,1].reshape(-1,1))[:,1]

# apply the optimum threshold obtained from the uncalibrated model here
custom_y_pred1_platts_highsensitivity_opt= np.where(custom_y_pred1_platts > 0.9, 1, 0)
print(custom_y_pred1_platts_highsensitivity_opt)

#%%
#compute performance metrics after thresholding the Platt scaling

mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_platts_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%
#Isotonic regression

iso = IsotonicRegression(out_of_bounds = 'clip')
iso.fit(custom_y_pred1, Y_test)

#perform isotonic regression
custom_y_pred1_isotonic = iso.predict(custom_y_pred1)

# apply the optimum threshold obtained from the uncalibrated model here
custom_y_pred1_isotonic_highsensitivity_opt= np.where(custom_y_pred1_isotonic > 0.9, 1, 0)
print(custom_y_pred1_isotonic_highsensitivity_opt)

#%%
#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_isotonic_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%

# Beta calibration
bc = BetaCalibration()
bc.fit(custom_y_pred1, Y_test)

#perform beta calibration
custom_y_pred1_beta = bc.predict(custom_y_pred1)

# apply the optimum threshold obtained from the uncalibrated model here
custom_y_pred1_beta_highsensitivity_opt= np.where(custom_y_pred1_beta > 0.9, 1, 0)
print(custom_y_pred1_beta_highsensitivity_opt)

#%%
#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_beta_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%
# Spline calibration
splinecalib = mli.SplineCalib(penalty='l2',			
                              knot_sample_size=40,			
                              cv_spline=5,			
                              unity_prior=False,			
                              unity_prior_weight=128)	

# fit on the uncalibrated prediction
splinecalib.fit(custom_y_pred1, Y_test)

#perform Spline calibration
custom_y_pred1_spline = splinecalib.predict(custom_y_pred1)

# apply the optimum threshold obtained from the uncalibrated model here
custom_y_pred1_spline_highsensitivity_opt= np.where(custom_y_pred1_spline > 0.9, 1, 0)
print(custom_y_pred1_spline_highsensitivity_opt)

#%%
#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_spline_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%
# Ensemble of 3 methods

# -----------------------------------------------------Platts
lr = LogisticRegression(C=99999999999, solver='lbfgs')
lr.fit(custom_y_pred1.reshape(-1,1), Y_test)
custom_y_pred1_caliberatedtemp = lr.predict_proba(custom_y_pred1.reshape(-1,1))[:,1]
y_value_platt, x_value_platt = calibration_curve(Y_test, custom_y_pred1_caliberatedtemp, n_bins=10)

#------------------------------------------------------ Beta
bc = BetaCalibration()
bc.fit(custom_y_pred1, Y_test)
custom_y_pred1_caliberatedtemp = bc.predict(custom_y_pred1)
y_value_beta, x_value_beta = calibration_curve(Y_test, custom_y_pred1_caliberatedtemp, n_bins=10)

#-----------------------------------------------------Spline
splinecalib = mli.SplineCalib(penalty='l2',			
                              knot_sample_size=40,			
                              cv_spline=5,			
                              unity_prior=False,			
                              unity_prior_weight=128)
splinecalib.fit(custom_y_pred1, Y_test)
custom_y_pred1_caliberatedtemp = splinecalib.predict(custom_y_pred1)
y_value_spline, x_value_spline = calibration_curve(Y_test, custom_y_pred1_caliberatedtemp, n_bins=10)

#%%
# choose the values with the least difference between accuracy and confidence to create ensemble

ensemblechoice = []
for i in range(10):
  currentx = x_value_platt[i]
  currenty = y_value_platt[i]
  mse1 = (currenty - currentx)**2

  currentx = x_value_beta[i]
  currenty = y_value_beta[i]
  mse2 = (currenty - currentx)**2

  currentx = x_value_spline[i]
  currenty = y_value_spline[i]
  mse3 = (currenty - currentx)**2

  least_error = np.argmin((mse1, mse2, mse3))
  ensemblechoice.append(least_error)
  print((i,least_error))

#%%

# create Ypred from ensemble
custom_y_pred1_caliberatedensemble=[]
bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range(len(custom_y_pred1)):
  bin_index = np.digitize(custom_y_pred1[i], bins)
  whatmethod = ensemblechoice[bin_index-1]
  if whatmethod==0:
    newpred = lr.predict_proba(custom_y_pred1[i].reshape(-1,1))[:,1]
    
    custom_y_pred1_caliberatedensemble.append(newpred)
    #print(newpred)
  elif whatmethod==1:
    newpred = bc.predict(custom_y_pred1[i].reshape(-1,1))
    custom_y_pred1_caliberatedensemble.append(newpred)
    #print(newpred)
  elif whatmethod==2:
    newpred = splinecalib.predict(custom_y_pred1[i].reshape(-1,1))
    custom_y_pred1_caliberatedensemble.append(newpred)
    #print(newpred)

custom_y_pred1_caliberatedensemble = np.array(custom_y_pred1_caliberatedensemble)
print(custom_y_pred1_caliberatedensemble)

#%%
# apply the optimum threshold obtained from the uncalibrated model here
custom_y_pred1_caliberatedensemble_highsensitivity_opt= np.where(custom_y_pred1_caliberatedensemble > 0.9, 1, 0)
print(custom_y_pred1_caliberatedensemble_highsensitivity_opt)

#%%
#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_caliberatedensemble_highsensitivity_opt,
                      beta=0.4)
print (mat_met)

#%%