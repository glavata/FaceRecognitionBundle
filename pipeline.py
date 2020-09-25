import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from components.preprocessor import Preprocessor
from components.classifier import Classifier
from time import perf_counter 
from statistics import mean
from pathlib import Path

class Pipeline:


    def __init__(self, X, y, z, dataset_name, preprocessor, classifier):
        self.__X = X
        self.__y = y
        self.__z = z
        self.__dataset_name = dataset_name
        self.__prepro = preprocessor
        self.__classifier = classifier

        if not os.path.exists('temp_data'):
            os.makedirs('temp_data')

        if not os.path.exists('temp_data/batches'):
            os.makedirs('temp_data/batches')


    def train(self, folds = 10, num_epochs_pre = 1, batch_count_pre = 1, num_epochs_post = 1, batch_count_post = 1):
        
        self.__folds = folds
        times_split = []

        #labels = list(range(0,self.__z))
        conf_mat = np.zeros((self.__z,self.__z))
        acc_s = 0

        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        i = 0

        self.__clear_folder('temp_data/')
        self.__clear_folder('temp_data/batches/')

        for train_split, test_split in skf.split(self.__X, self.__y):
            self.__save_train_test_split(train_split, test_split)
        
            if(self.__prepro != None):
                self.__train_epoch_split(num_epochs_pre, i, batch_count_pre, True, True)

            t_start_train_elapsed = perf_counter()
            self.__train_epoch_split(num_epochs_post, i, batch_count_post, False, True)
            t_end_train_elapsed = perf_counter()

            times_split.append(t_end_train_elapsed - t_start_train_elapsed)

            post_X_test, post_y_test = self.__load_split(train = False)
            if(self.__prepro != None):
                post_X_test = self.__prepro.get_out_data(post_X_test)

            y_pred = self.__classifier.predict(post_X_test)

            acc_s += accuracy_score(post_y_test,y_pred)
            #res_c = confusion_matrix(post_y_test, y_pred, labels=labels)
            #conf_mat += res_c

            i+=1
            self.__prepro.reinit()
            self.__classifier.reinit()

    
        acc_s /= folds
        
        time_mins_avg_train_elapsed = mean(times_split)

        return [conf_mat, acc_s, time_mins_avg_train_elapsed]


    def __train_epoch_split(self, num_epochs, tr_split_ind, batch_count, preproc, shuffle = False):
        
        for e in range(0,num_epochs):
            X_train, y_train = self.__load_split(train = True)
            #self.__clear_folder('temp_data/batches')

            if(shuffle):
                rand_ind = np.array(list(range(0,X_train.shape[0])))
                np.random.shuffle(rand_ind)

                X_train = X_train[rand_ind]
                y_train = y_train[rand_ind]

            if(batch_count > 1):
                skf = KFold(n_splits=batch_count, shuffle=False)
                s = 0
                #save train x/y split in parts(batches)
                for _, part_ind in skf.split(X_train, y_train):
                    X_part = X_train[part_ind]
                    y_part = y_train[part_ind]
                    self.__save_data_generic('temp_data/batches/' + self.__dataset_name + '_' + str(s), X_part, y_part)
                    s+=1
            else:
                self.__save_data_generic('temp_data/batches/' + self.__dataset_name + '_0' , X_train, y_train)

            for j in range(0, batch_count):
                #open train x/y split part(batch) and train
                X_batch, y_batch = self.__load_data_generic('temp_data/batches/' + self.__dataset_name + '_' + str(j))
                target = self.__prepro
                text = "PREPROCESSING: "
                if(preproc == False):
                    text = "TRAINING:      "
                    target = self.__classifier
                    if(self.__prepro != None):
                        X_batch = self.__prepro.get_out_data(X_batch)
                res = target.train_model(X_batch, y_batch)

                print("{0} val_split {1}/{2} epoch {3}/{4} batch {5}/{6}  ".format(text, tr_split_ind+1, self.__folds, \
                                                                                e+1, num_epochs, j+1,batch_count), \
                                                                                     end="\r", flush=True)
            
        
    def __clear_folder(self, folder):
        for parent, dirnames, filenames in os.walk(folder):
            for fn in filenames:
                os.remove(os.path.join(parent, fn))

    def __save_train_test_split(self, train_split, test_split):
        #save train x/y split
        with open('temp_data/' + self.__dataset_name + '_train_split', 'wb') as f:
            X_train, y_train = self.__X[train_split], self.__y[train_split]
            np.savez(f, X_train, y_train)

        #save test x/y split
        with open('temp_data/' + self.__dataset_name + '_test_split', 'wb') as f:
            X_test, y_test = self.__X[test_split], self.__y[test_split]
            np.savez(f, X_test, y_test)

    def __load_split(self, train):
        subfol = '_train_split' if train else '_test_split'
        X, y = self.__load_data_generic('temp_data/' + self.__dataset_name + subfol)

        return X, y

    def __load_data_generic(self, folder):
        res1, res2 = None, None
        with open(folder, 'rb') as f:
            rdy_arr = np.load(f, allow_pickle=True)
            res1, res2 = rdy_arr['arr_0'], rdy_arr['arr_1']

        return res1, res2

    def __save_data_generic(self, dir_tgt, part1, part2):
        with open(dir_tgt, 'wb') as f:
            np.savez(f, part1, part2)