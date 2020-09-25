import os
from components.cnn_svm import CNN_SVM
from components.classifiers.svm_c import SVM_C
from time import perf_counter 
from data_load import DataLoader, Dataset
from pipeline import Pipeline
import numpy as np
import atexit

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def savecounter(filename, arr1, arr2, arr3):
    with open(filename, 'wb') as fl:
        np.savez(fl, arr1, arr2, arr3)



import seaborn as sn
import matplotlib.pyplot as plt

#cur_dataset = Dataset.TU
#dataloader = DataLoader(cur_dataset, {'angle_limit':10, 'img_format':None})

cur_dataset = Dataset.YALE_EX_CROPPED
dataloader = DataLoader(cur_dataset, {'resize':True})
X,y,z,v = dataloader.load_data(reload=False) 

#X = X/255.0
#z is the number of classes
#v is the map of labels to which 0-num_classes correspond

num_val_splits = 10
#Find optimal parameters CNN

X_shape =  (X.shape[1], X.shape[2])


# #CNN params 
# epochs_arr = [2]
# batch_count_arr = [10,20,30]
# conv_layers_count_arr = [1,2,3]
# conv_filters_arr = [[3,3,3], [3,5,7],[3,5,3]]
# conv_filters_count = [[12,16,20]]#[[4,8,12],[8,12,16]]
# first_dense_layer = [1024, 512, 256]
# final_feat_vec = [None, 32,64,128,256]

# total_len = len(epochs_arr) * len(batch_count_arr) * len(conv_layers_count_arr) * \
#     len(conv_filters_arr) * len(conv_filters_count) * len(first_dense_layer) * \
#         len(final_feat_vec)
    

#SVM params
kernel_arr = ['linear'] #['linear','poly','gauss']
gamma_arr = [0.1, 0.5, 1, 3 ,6] 
degree_arr  = [0.1, 0.5, 1, 1.2, 1.5, 2, 4, 6, 8]
C_arr = [0.001, 0.01, 0.1, 1, 5, 10, 30]

total_len = len(kernel_arr) * len(gamma_arr) * len(degree_arr) * len(C_arr)

acc_s = []
times = []
params_r = []

t_start_total = perf_counter()  

i = 0

for a in kernel_arr:
    for b in gamma_arr:
        for c in degree_arr:
            for d in C_arr:

                obj_set = {'Kernel': a, 'C': b, 'Degree': c, 'Gamma': d}
                print("current config {0}/{1} - {2}".format(i+1,total_len,str(obj_set)))

                cnn_s = CNN_SVM({'RGB':False, \
                'ConvCount': 2, \
                'ConvFilterSizes':[3,3,3], \
                'ConvFilterCount':[12,16,20], \
                'FirstDenseLayer':512, \
                'FinalFeatVec':None
                }, z, X_shape)
                svm_c = SVM_C(obj_set)
                try:
                    pipeline = Pipeline(X, y, z, cur_dataset.name, cnn_s, svm_c)
                    results = pipeline.train(num_val_splits, 2, 30, 1, 1)
                except Exception as ex:
                    print(str(ex))
                    with open("results/reserror_cur", 'wb') as fl:
                        np.savez(fl, acc_s, times, params_r)

                t_cur_total = perf_counter()
                t_cur_total_elapsed = t_cur_total - t_start_total

                print('cur time elapsed: {0:.2f} hours, {1:.2f} mins, {2:.2f} secs'.format(t_cur_total_elapsed/60/60, t_cur_total_elapsed/60, t_cur_total_elapsed))

                acc_s.append(results[1])
                times.append(results[2])
                params_r.append(obj_set)
                i+=1  


with open("results/res_cur", 'wb') as fl:
     np.savez(fl, acc_s, times, params_r)


#sn.heatmap(results[0], annot=True, annot_kws={"size": 10})
#plt.show()