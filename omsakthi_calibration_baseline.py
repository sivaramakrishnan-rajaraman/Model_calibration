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
   beta = 0.4
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
# #save the predictions
# np.savetxt('vgg16_cardiomegaly_y_pred.csv',custom_y_pred,fmt='%f',delimiter = ",")
# np.savetxt('Y_test.csv',Y_test,fmt='%f',delimiter = ",")

#%%

#we need the scores of only the positive abnormal class
custom_y_pred1 = custom_y_pred[:,1]

#%%
#print all metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#using seaborn print confusion matrix
target_names = ['Abnormal','Normal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
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
#compute ECE of the baseline model
print('The expected calibration error of the trained model is' , 
      round(ece_score(custom_y_pred,Y_test,n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,custom_y_pred[:,1]),4))

#compute Log loss
print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,custom_y_pred[:,1]),4))

#%%
# plot reliability diagrams for the baseline model
rd = mli.plot_reliability_diagram(Y_test,custom_y_pred[:,1])

# with histograms
mli.plot_reliability_diagram(Y_test,custom_y_pred[:,1], 
                             show_histogram=True)

#using custom bins
plt.figure(figsize=(10,5), dpi=400)
custom_bins_a = np.array([0,.01,.02,.03,.05, .1, .3, .5, .75, 1])
mli.plot_reliability_diagram(Y_test,custom_y_pred[:,1], 
                             bins=custom_bins_a, 
                             show_histogram=True)

#%%
#plot calibration curve for the baseline uncalibrated model

labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated model')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

'''
The calibration curve is entirely above the reference line of a calibration plot 
is called “underprediction,” if entirely below, it is called "overprediction"
'''

#%%
'''
Calibrating the scores
We can clearly see from the histogram that the probabilities 
produced by the model tend to assume extreme values. Therefore, 
it might be useful to apply calibration techniques to try and fix these 
distortions. Two calibration methods have been widely used in machine 
learning literature: logistic calibration and isotonic regression. 
The first one is a parametric method that assumes an underlying distribution 
of the scores composed of two Gaussians of equal variance, 
(one for the positives and another for the negatives). 

The second method is a non-parametric approach, therefore it doesn't make
any assumption about the distribution of the scores, however, it needs 
lots of data to produce a good model. Let's see the effect of applying 
these methods to the previously trained classifier's outputs.

Platt scaling:
methods of calibration: Logistic regression (Platt scaling)
In Platt’s case, we are essentially just performing logistic 
regression on the output of the DL model (y_pred) with respect to the true class labels (Y_test).
'''

#%%
# Fit Platt scaling (logistic calibration)
lr = LogisticRegression(C=99999999999, solver='lbfgs')
lr.fit(custom_y_pred[:,1].reshape(-1,1), Y_test)
custom_y_pred1_platts1 = lr.predict_proba(custom_y_pred[:,1].reshape(-1,1))
custom_y_pred1_platts = lr.predict_proba(custom_y_pred[:,1].reshape(-1,1))[:,1]

#Reliability Diagram on Test Data after Platt Calibration
mli.plot_reliability_diagram(Y_test, custom_y_pred1_platts)
plt.title('Reliability Diagram on Test Data after Platt Calibration')

#%%
#plot calibration curves together
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated model')
frac_of_positives_platt, pred_prob_platt = calibration_curve(Y_test, custom_y_pred1_platts, n_bins=10)
plt.plot(pred_prob_platt,
         frac_of_positives_platt,
         linewidth=4,
         color='blue')
labels.append('Platt scaling')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#compute performance metrics after Platt scaling

mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_platts1.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#compute ECE
print('The expected calibration error of the Platt-scaled calibrated model is' , 
      round(ece_score(custom_y_pred1_platts1, Y_test, n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the trained model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1_platts),4))

#compute Log loss
print('The Log Loss of the trained model is' , 
      round(log_loss(Y_test,custom_y_pred1_platts),4))

#%%
# plot roc curve and find the optimum threshold

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred1_platts)

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred1_platts)
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
         label='Platt-scaled model')

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
AUC: 0.9577718065316246
Best Threshold=0.950308, G-Mean=0.894
Best Threshold=0.950308, Youden J statistic=0.789
'''

#%%
'''
Isotonic regression:
Fits a piecewise constant, monotonically increasing, function to map the scores to probabilities.
Uses the PAV (Pool Adjacent Violators, also called PAVA) algorithm.
Does not assume a particular parametric form.
Tends to be better than Platt scaling with enough data
Tends to overfit in the case of sparse data: ("choppy" with unrealistic jumps)
Reference: Zadrozny, B., & Elkan, C.(2001). Obtaining calibrated probability estimates 
from decision trees and naive bayesian classifiers. ICML (pp.609–616).

Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate
multiclass probability estimates. KDD (pp.694–699).
out_of_bounds{‘nan’, ‘clip’, ‘raise’}, default=’nan’
Handles how X values outside of the training domain are handled during prediction.
'''
iso = IsotonicRegression()
#iso = IsotonicRegression(out_of_bounds = 'clip')
iso.fit(custom_y_pred1, Y_test)

#perform isotonic regression
custom_y_pred1_isotonic = iso.predict(custom_y_pred1)

#%%
#plot Isotonic Calibration Curve on Test Data
mli.plot_reliability_diagram(Y_test, custom_y_pred1, error_bars=False)
tvec = np.linspace(.01, .99, 99)
plt.plot(tvec, iso.predict(tvec), label='Isotonic regression')
plt.title('Isotonic Calibration Curve on Test Data')

#%%
#Reliability Diagram on Test Data after Isotonic Calibration
mli.plot_reliability_diagram(Y_test, custom_y_pred1_isotonic)
plt.title('Reliability Diagram on Test Data after Isotonic Calibration')

#%%
#plot calibration curves together
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated model')
frac_of_positives_platt, pred_prob_platt = calibration_curve(Y_test, custom_y_pred1_platts, n_bins=10)
plt.plot(pred_prob_platt,
         frac_of_positives_platt,
         linewidth=4,
         color='blue')
labels.append('Platt scaling')
frac_of_positives_isotonic, pred_prob_isotonic = calibration_curve(Y_test,custom_y_pred1_isotonic, n_bins=10)
plt.plot(pred_prob_isotonic,
         frac_of_positives_isotonic,
         linewidth=4,
         color='yellow')
labels.append('Isotonic regression')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#compute performance metrics after Platt scaling

custom_y_pred1_isotonic_1 = np.array(1-custom_y_pred1_isotonic)
custom_y_pred1_isotonic_2 = np.c_[custom_y_pred1_isotonic_1, custom_y_pred1_isotonic]

#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_isotonic_2.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#compute ECE
print('The expected calibration error of the Platt-scaled calibrated model is' , 
      round(ece_score(custom_y_pred1_isotonic_2, Y_test, n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the Platt-scaled calibrated model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1_isotonic),4))

#compute Log loss
print('The Log Loss of the Platt-scaled calibrated model is' , 
      round(log_loss(Y_test,custom_y_pred1_isotonic),4))


#%%
# plot roc curve and find the optimum threshold

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred1_isotonic)

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred1_isotonic)
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
         label='Isotonic regression')

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
AUC: 0.9605105415460935
Best Threshold=0.905172, G-Mean=0.894
Best Threshold=0.905172, Youden J statistic=0.789
'''
#%%
'''
Method 3: Beta Calibration
"A well-founded and easily implemented improvement on logistic calibration for binary classifiers."

$p = \left(1+ 1 / \left( \exp(c) \frac{z^a}{(1-z)^b} \right) \right)^{-1}$

Similar to Platt scaling with a couple of important improvements
Is a 3-parameter family of curves rather than 2-parameter
Family of curves includes the line $y=x$ (so it won't mess it up if it's already calibrated)
Reference: Kull, M., Filho, T.S. & Flach, P.. (2017). Beta calibration: a well-founded and 
easily implemented improvement on logistic calibration for binary classifiers. 
Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, in PMLR 54:623-631

For optimal decision making under variable class distributions and misclassification 
costs a classifier needs to produce well-calibrated estimates of the posterior probability. 
Isotonic calibration is a powerful non-parametric method that is however prone to 
overfitting on smaller datasets; hence a parametric method based on the logistic 
curve is commonly used. While logistic calibration is designed to correct for a 
specific kind of distortion where classifiers tend to score on too narrow a scale, 
we demonstrate experimentally that many classifiers including naive Bayes and Adaboost 
suffer from the opposite distortion where scores tend too much to the extremes. 
In such cases logistic calibration can easily yield probability estimates that are 
worse than the original scores. Moreover, the logistic curve family does not include 
the identity function, and hence logistic calibration can easily uncalibrate a perfectly calibrated classifier.

In this paper we solve all these problems with a richer class of 
calibration maps based on the Beta distribution. We derive the method from 
first principles and show that fitting it is as easy as fitting a logistic curve. 
Extensive experiments show that beta calibration is superior to logistic calibration for naive Bayes and Adaboost.
'''
#%%
# Fit three-parameter beta calibration
bc = BetaCalibration()
bc.fit(custom_y_pred1, Y_test)

#perform beta calibration
custom_y_pred1_beta = bc.predict(custom_y_pred1)

#plot Beta calibration surve on test set
mli.plot_reliability_diagram(Y_test, custom_y_pred1_beta, error_bars=False)
tvec = np.linspace(.01, .99, 99)
plt.plot(tvec, bc.predict(tvec))
plt.title('Beta Calibration Curve on Test Set')

# Reliability Diagram on Test Data after Beta Calibration
mli.plot_reliability_diagram(Y_test, custom_y_pred1_beta)
plt.title('Reliability Diagram on Test Data after Beta Calibration')

#%%
#plot calibration curves together
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated model')
frac_of_positives_platt, pred_prob_platt = calibration_curve(Y_test, custom_y_pred1_platts, n_bins=10)
plt.plot(pred_prob_platt,
         frac_of_positives_platt,
         linewidth=4,
         color='blue')
labels.append('Platt scaling')
frac_of_positives_beta, pred_prob_beta = calibration_curve(Y_test,custom_y_pred1_beta, n_bins=10)
plt.plot(pred_prob_beta,
         frac_of_positives_beta,
         linewidth=4,
         color='yellow')
labels.append('Beta calibration')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#compute performance metrics after Platt scaling

custom_y_pred1_beta_1 = np.array(1-custom_y_pred1_beta)
custom_y_pred1_beta_2 = np.c_[custom_y_pred1_beta_1, custom_y_pred1_beta]

#performance metrics
mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_beta_2.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#compute ECE
print('The expected calibration error of the Beta calibrated model is' , 
      round(ece_score(custom_y_pred1_beta_2, Y_test, n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the Beta calibrated model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1_beta),4))

#compute Log loss
print('The Log Loss of the Beta calibrated model is' , 
      round(log_loss(Y_test,custom_y_pred1_beta),4))

#%%
# plot roc curve and find the optimum threshold

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred1_beta)

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred1_beta)
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
         label='Beta calibration')

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
0.9577718065316246
Best Threshold=0.890313, G-Mean=0.894
Best Threshold=0.890313, Youden J statistic=0.789
'''
#%%
'''
Method 4: SplineCalib
SplineCalib fits a cubic smoothing spline to the relationship between the 
uncalibrated scores and the calibrated probabilities
Smoothing splines strike a balance between fitting the points well and having a smooth function
SplineCalib uses a smoothed logistic function - so the fit to data is measured by 
likelihood (i.e. log-loss) and the smoothness refers to the integrated second derivative before the logistic transformation.
There is a nuisance parameter that trades off smoothness for fit. At one extreme 
it will revert to standard logistic regression (i.e. Platt scaling) and at the other extreme it will be a very wiggly function that fits the data but does not generalize well.

SplineCalib automatically fits the nuisance parameter (though this can be adjusted by the user)

The resulting calibration function is not necessarily monotonic. (In some cases this may be beneficial).
References: Lucena, B. Spline-based Probability Calibration. https://arxiv.org/abs/1809.07751

'''
#%%
# Define SplineCalib object

splinecalib = mli.SplineCalib(penalty='l2',			
                              knot_sample_size=40,			
                              cv_spline=5,			
                              unity_prior=False,			
                              unity_prior_weight=128)	
		
# fit on the uncalibrated prediction
splinecalib.fit(custom_y_pred1, Y_test)

#perform Spline calibration
custom_y_pred1_spline = splinecalib.predict(custom_y_pred1)
custom_y_pred1_spline_1 = np.array(1-custom_y_pred1_spline)
custom_y_pred1_spline_2 = np.c_[custom_y_pred1_spline_1, custom_y_pred1_spline]

#%%
#plot Spline calibration surve on test set
mli.plot_reliability_diagram(Y_test, custom_y_pred1_spline, 
                             error_bars=False)
tvec = np.linspace(.01, .99, 99)
plt.plot(tvec, splinecalib.predict(tvec))
plt.title('Spline Calibration Curve on Test Set')

#%%
# Reliability Diagram on Test Data after Spline Calibration
mli.plot_reliability_diagram(Y_test, custom_y_pred1_spline)
plt.title('Reliability Diagram on Test Data after Spline Calibration')

#%%
#plot calibration curves together
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated')
frac_of_positives_platt, pred_prob_platt = calibration_curve(Y_test, custom_y_pred1_platts, n_bins=10)
plt.plot(pred_prob_platt,
         frac_of_positives_platt,
         linewidth=4,
         color='blue')
labels.append('Platt scaling')
frac_of_positives_beta, pred_prob_beta = calibration_curve(Y_test,custom_y_pred1_beta, n_bins=10)
plt.plot(pred_prob_beta,
         frac_of_positives_beta,
         linewidth=4,
         color='yellow')
labels.append('Beta calibration')
frac_of_positives_spline, pred_prob_spline = calibration_curve(Y_test,custom_y_pred1_spline, n_bins=10)
plt.plot(pred_prob_spline,
         frac_of_positives_spline,
         linewidth=4,
         color='red')
labels.append('Spline calibration')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#compute performance metrics after spline calibration

mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_spline_2.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#compute ECE
print('The expected calibration error of the Spline calibrated model is' , 
      round(ece_score(custom_y_pred1_spline_2, Y_test, n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the Spline calibrated model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1_spline),4))

#compute Log loss
print('The Log Loss of the Spline calibrated model is' , 
      round(log_loss(Y_test,custom_y_pred1_spline),4))

#%%
# plot roc curve and find the optimum threshold

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred1_spline)

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred1_spline)
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
         label='Spline calibration')

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
0.9577718065316246
Best Threshold=0.887578, G-Mean=0.894
Best Threshold=0.887578, Youden J statistic=0.789
'''

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
#draw reliability diagram with the ensemble prediction

custom_y_pred1_caliberatedtemp = custom_y_pred1_caliberatedensemble[:,0]
vartemp = np.array(1-custom_y_pred1_caliberatedtemp)
custom_y_pred1_caliberated = np.c_[vartemp, custom_y_pred1_caliberatedtemp]

#%%
#Reliability Diagram on Test Data after Calibration
mli.plot_reliability_diagram(Y_test, custom_y_pred1_caliberatedtemp)
plt.title('Reliability Diagram on Test Data after Calibration')

#%%
#plotting only for ensemble
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_ensemble, pred_prob_ensemble = calibration_curve(Y_test,custom_y_pred1_caliberatedtemp, n_bins=10)
plt.plot(pred_prob_ensemble,
         frac_of_positives_ensemble,
         linewidth=4,
         color='magenta')
labels.append('Ensemble calibration')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#plot calibration curves together
labels = []
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.10) 
minor_ticks = np.arange(0.0, 1.1, 0.10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
frac_of_positives_uncalibrated, pred_prob_uncalibrated = calibration_curve(Y_test,custom_y_pred1, n_bins=10)
plt.plot(pred_prob_uncalibrated,
         frac_of_positives_uncalibrated,
         linewidth=4,
         color='green')
labels.append('Uncalibrated')
frac_of_positives_platt, pred_prob_platt = calibration_curve(Y_test, custom_y_pred1_platts, n_bins=10)
plt.plot(pred_prob_platt,
         frac_of_positives_platt,
         linewidth=4,
         color='blue')
labels.append('Platt scaling')
frac_of_positives_beta, pred_prob_beta = calibration_curve(Y_test,custom_y_pred1_beta, n_bins=10)
plt.plot(pred_prob_beta,
         frac_of_positives_beta,
         linewidth=4,
         color='yellow')
labels.append('Beta calibration')
frac_of_positives_spline, pred_prob_spline = calibration_curve(Y_test,custom_y_pred1_spline, n_bins=10)
plt.plot(pred_prob_spline,
         frac_of_positives_spline,
         linewidth=4,
         color='red')
labels.append('Spline calibration')
frac_of_positives_ensemble, pred_prob_ensemble = calibration_curve(Y_test,custom_y_pred1_caliberatedtemp, n_bins=10)
plt.plot(pred_prob_ensemble,
         frac_of_positives_ensemble,
         linewidth=4,
         color='magenta')
labels.append('Ensemble calibration')
plt.plot([0, 1], [0, 1], color='black', 
         linestyle='dashed', 
         linewidth = 1)
labels.append('Perfectly calibrated')
plt.legend(labels,loc="lower right", prop={"size":20})
xlabel = plt.xlabel("Average Probability")
ylabel = plt.ylabel("Fraction of positives")
plt.show()

#%%
#compute performance metrics after spline calibration

mat_met = perf_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred1_caliberated.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%
#compute ECE
print('The expected calibration error of the Ensemble calibrated model is' , 
      round(ece_score(custom_y_pred1_caliberated, Y_test, n_bins=10),4))

#compute Brier score loss
print('The Brier Score Loss of the Ensemble calibrated model is' , 
      round(brier_score_loss(Y_test,custom_y_pred1_caliberatedtemp),4))

#compute Log loss
print('The Log Loss of the Ensemble calibrated model is' , 
      round(log_loss(Y_test,custom_y_pred1_caliberatedtemp),4))

#%%
# plot roc curve and find the optimum threshold

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred1_caliberatedtemp)

#compute area under the ROC curve
auc_score=roc_auc_score(Y_test, custom_y_pred1_caliberatedtemp)
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
         label='Ensemble calibration')

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
0.9574766891736715
Best Threshold=0.887578, G-Mean=0.894
Best Threshold=0.887578, Youden J statistic=0.789
'''
#%%