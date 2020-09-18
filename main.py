import os
from components.cnn_svm import CNN_SVM
from components.classifiers.svm_c import SVM_C
from time import perf_counter 
from data_load import DataLoader, Dataset
from pipeline import Pipeline
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import seaborn as sn
import matplotlib.pyplot as plt

#cur_dataset = Dataset.TU
#dataloader = DataLoader(cur_dataset, {'angle_limit':10, 'img_format':None})

cur_dataset = Dataset.YALE_EX_CROPPED
dataloader = DataLoader(cur_dataset)
X,y,z,v = dataloader.load_data(reload=False) 

#X = X/255.0
#z is the number of classes
#v is the map of labels to which 0-num_classes correspond

num_val_splits = 10
#Find optimal parameters CNN

X_shape =  (X.shape[1], X.shape[2])


#CNN params 
epochs_arr = [2,3]
batch_count_arr = [10,20,30]
conv_layers_count_arr = [1,2,3]
conv_filters_arr = [[3,5,7],[3,3,3],[3,5,3]]
conv_filters_count = [[4,8,12],[8,12,16],[12,16,20]]
first_dense_layer = [1024, 512, 256]
final_feat_vec = [32,64,128,256, None]

#SVM params
kernel_arr = ['linear','poly','gauss']
gamma_arr = [0.1, 0.5, 1, 3 ,6] 
degree_arr  = [0.1, 0.5, 1, 1.2, 1.5, 2, 4, 6, 8]
C_arr = [0.001, 0.01, 0.1, 1, 5, 10, 30]

acc_s = []
times = []
params_r = []

for a in epochs_arr:
    for b in batch_count_arr:
        for c in conv_layers_count_arr:
            for d in conv_filters_arr:
                for e in conv_filters_count:
                    for f in first_dense_layer:
                        for g in final_feat_vec:
                            cnn_s = CNN_SVM({'RGB':False, \
                                            'ConvCount': c, \
                                            'ConvFilterSizes':d, \
                                            'ConvFilterCount':e, \
                                            'FirstDenseLayer':f, \
                                            'FinalFeatVec':g
                                            }, z, X_shape)

                            pipeline = pipeline(X, y, z, cur_dataset.name, None, cnn_s)
                            results = pipeline.train(num_val_splits, a, b)
                            acc_s.append(results[1])
                            times.append(results[2])
                            params_r.append({'epochs': a, 'batch_count': b, 'ConvCount': c, 'ConvFilterSizes': str(d), \
                                            'ConvFilterCount': str(e), 'FirstDenseLayer':f, 'FinalFeatVec':g})     



# svm_c = svm_c({'kernel':'poly', \
#                 'c':0.1, \
#                 'degree':1.5, \
#                 'gamma':6.8
#                 }, z, x_shape, true)

#sn.heatmap(results[0], annot=True, annot_kws={"size": 10})
#plt.show()

with open("results/res1", 'wb') as f:
    np.savez(f, acc_s, times, params_r)


