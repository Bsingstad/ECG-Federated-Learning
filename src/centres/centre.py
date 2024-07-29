from src.models.model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os


class LocalHospital:
    def __init__(self, name, X_path, y_path):
        self.name = name
        self.X_data, self.y_data = np.load(X_path), np.load(y_path)
        self.X_data = np.moveaxis(self.X_data,1,-1)
        self.val_auroc = []
        print(self.X_data.shape)
        print(self.y_data.shape)
        self.train_val_test_split()

    def load_model(self, model):
        self.model = model

    def train_val_test_split(self):
        X_train_temp, self.X_test, y_train_temp, self.y_test = train_test_split(self.X_data, self.y_data, test_size=0.20, random_state=42) #20% test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1875, random_state=42) #15% val

    def train_one_epoch(self):
        batch_size = 32
        if self.X_train.shape[0] < 100:
            batch_size = 10 # an exeption for very small dataset
        self.model.fit(self.X_data,self.y_data, epochs=1, batch_size = batch_size, 
                       steps_per_epoch = self.X_train.shape[0]//batch_size)

    def predict_val(self):
        val_hat = self.model.predict(self.X_val)
        self.val_auroc.append(roc_auc_score(self.y_val, val_hat))

    def predict_test(self):
        test_hat = self.model.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test.ravel(), test_hat.ravel())
        test_auroc = roc_auc_score(self.y_test.ravel(), test_hat.ravel())
        
        #plt.figure()
        #plt.plot(fpr, tpr, label='micro-average ROC curve (area = {0:0.2f})'''.format(test_auroc))
        #plt.show()

        return fpr, tpr, test_auroc
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train_to_convergence(self):
        my_callbacks = []
        batch_size = 32
        if self.X_train.shape[0] < 100:
            batch_size = 10 # an exeption for very small dataset
        monitor_variable = "val_AUROC"
        monitor_mode = "max"
        temp_model_folder = "./models/temp_folder/"
        temp_model_name = "model_weights_temp.h5"
        os.makedirs(temp_model_folder, exist_ok=True)
        my_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_variable, mode=monitor_mode, patience=5, verbose=1, restore_best_weights=True))
        my_callbacks.append(tf.keras.callbacks.ModelCheckpoint(temp_model_folder + temp_model_name, monitor=monitor_variable, mode=monitor_mode, save_best_only=True, verbose=1))
        self.model.fit(self.X_train, self.y_train, epochs = 100, batch_size = batch_size, steps_per_epoch = self.X_train.shape[0]//batch_size,
                       validation_data=(self.X_val, self.y_val), validation_steps = self.X_val.shape[0]//batch_size, callbacks=my_callbacks, shuffle=True)

        self.model.load_weights(temp_model_folder + temp_model_name)
        shutil.rmtree(temp_model_folder)
        
        print("Training complete")

    @staticmethod
    def batch_generator(batch_size, gen_data): 
        batch_features = np.zeros((batch_size,1000, 12))
        batch_labels = np.zeros((batch_size,30)) #drop undef class
        while True:
            for i in range(batch_size):
                batch_features[i], batch_labels[i] = next(gen_data) 
            yield batch_features, batch_labels

    
    staticmethod
    def generate_data(X,y):
        while True:
            for i in range(len(y)):
                y_out = y[i]
                X_out = X[i]
                yield X_out, y_out
    
    def set_generator(self):
        self.bgen = self.batch_generator(30,self.generate_data(self.X_train,self.y_train))
    
    def train_one_batch(self):
        loss_fn =tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
        X_batch, y_batch = next(self.bgen)
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients
    

class CentralModelDistributor:
    def __init__(self, input_shape, output_shape):
        self.model = build_iception_model(input_shape, output_shape)
        self.weight_list = []
        self.grads_list = []
        self.set_optimizer()
    
    def get_model(self):
        return self.model
    
    def set_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam()
        

    def update_weight_list(self, new_weights):
        self.weight_list.append(new_weights)

    def return_avg_weights(self):
        summed_weights = [sum(w) for w in zip(*self.weight_list)]
        avg_weight = [w / len(self.weight_list) for w in summed_weights]
        self.weight_list = []
        return avg_weight
    
    def load_weights_to_model(self, weights):
        self.model.set_weights(weights)

    def get_weights_from_model(self):
        return self.model.get_weights()
    
    def update_grads_list(self, new_grads):
        self.grads_list.append(new_grads)
    
    def apply_grads(self):
        summed_grads = [sum(g) for g in zip(*self.grads_list)]
        self.optimizer.apply_gradients(zip(summed_grads, self.model.trainable_variables))
        self.grads_list = []
    

class ExternalValidationHospital:
    def __init__(self, name, X_path, y_path):
        self.name = name
        self.X_data, self.y_data = np.load(X_path), np.load(y_path)
        self.val_auroc = []
        self.X_data = np.moveaxis(self.X_data,1,-1)
        print(self.X_data.shape)
        print(self.y_data.shape)
        self.train_val_test_split()

    def load_model(self, model):
        self.model = model
    
    def train_val_test_split(self):
        X_train_temp, self.X_test, y_train_temp, self.y_test = train_test_split(self.X_data, self.y_data, test_size=0.20, random_state=42) #20% test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.1875, random_state=42) #15% val

    def predict_val(self):
        val_hat = self.model.predict(self.X_val)
        self.val_auroc.append(roc_auc_score(self.y_val, val_hat))

    def predict_test(self):
        test_hat = self.model.predict(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test.ravel(), test_hat.ravel())
        test_auroc = roc_auc_score(self.y_test.ravel(), test_hat.ravel())
        
        plt.figure()
        plt.plot(fpr, tpr, label='micro-average ROC curve (area = {0:0.2f})'''.format(test_auroc))
        plt.show()

        return fpr, tpr, test_auroc
    
    def get_weights(self):
        return self.model.get_weights()
    
    def prepare_for_transfer_learning(self):
        self.model.trainable = False
        for layer in self.model.layers[-2]:
            layer.trainable = True
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train_to_convergence(self):
        my_callbacks = []
        monitor_variable = "val_AUROC"
        monitor_mode = "max"
        temp_model_folder = "./models/temp_folder/"
        temp_model_name = "model_weights_temp.h5"
        os.makedirs(temp_model_folder, exist_ok=True)
        my_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_variable, mode=monitor_mode, patience=5, verbose=1, restore_best_weights=True))
        my_callbacks.append(tf.keras.callbacks.ModelCheckpoint(temp_model_folder + temp_model_name, monitor=monitor_variable, mode=monitor_mode, save_best_only=True, verbose=1))
        self.model.fit(self.X_train, self.y_train, epochs = 100, batch_size = 32, steps_per_epoch = self.X_train.shape[0]//32,
                       validation_data=(self.X_val, self.y_val), validation_steps = self.X_val.shape[0]//32, callbacks=my_callbacks)

        self.model.load_weights(temp_model_folder + temp_model_name)
        shutil.rmtree(temp_model_folder)
        
        print("Training complete")


class CentralTrainer:
    def __init__(self, input_shape, output_shape):
        #self.X_data
        #self.y_data
        self.model = build_iception_model(input_shape, output_shape)
        self.switch = 0

    def load_data(self, X_path, y_path):
        print("Loading " + X_path)
        X_temp = np.load(X_path)
        X_temp = np.moveaxis(X_temp, 1,-1)
        y_temp = np.load(y_path)
        if self.switch == 0:
            self.X_data = X_temp
            self.y_data = y_temp
        else:
            self.X_data = np.vstack([self.X_data, X_temp])
            self.y_data = np.vstack([self.y_data, y_temp]) 
        self.switch = 1

    def train_val_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_data, self.y_data, test_size=0.15, random_state=42) #20% test

    def train_to_convergence(self):
        my_callbacks = []
        monitor_variable = "val_AUROC"
        monitor_mode = "max"
        temp_model_folder = "./models/temp_folder/"
        temp_model_name = "model_weights_temp.h5"
        os.makedirs(temp_model_folder, exist_ok=True)
        my_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor_variable, mode=monitor_mode, patience=5, verbose=1, restore_best_weights=True))
        my_callbacks.append(tf.keras.callbacks.ModelCheckpoint(temp_model_folder + temp_model_name, monitor=monitor_variable, mode=monitor_mode, save_best_only=True, verbose=1))
        self.model.fit(self.X_train, self.y_train, epochs = 100, batch_size = 32, steps_per_epoch = self.X_train.shape[0]//32,
                       validation_data=(self.X_val, self.y_val), validation_steps = self.X_val.shape[0]//32, callbacks=my_callbacks)

        self.model.load_weights(temp_model_folder + temp_model_name)
        shutil.rmtree(temp_model_folder)
        
        print("Training complete")

    def get_model(self):
        return self.model
    