import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, LSTM,Attention,Concatenate, Bidirectional,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping,Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,accuracy_score 
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt
import os

## CPU
num_cores = mp.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

##
def scale_data(data,scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    return scaled_data,scaler


### 
class PerObservationMSECallback(Callback):
    def __init__(self,train_dataset,val_dataset, folder_path = r'C:\Users\SESA626862\Documents\RL_codes\histogramas'):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.folder_path = folder_path
        self.mse_per_epoch_train = []
        self.mse_per_epoch_val = []

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def compute_mse(self,dataset,epoch):
        y_true_all = []
        y_pred_all = []
        X_=[]

        ## Create a new iterator for the dataset
        dataset_iterator = iter(dataset)

        ## Iterate over the dataset to collect all true and predicted values
        for X_batch,y_batch in dataset_iterator:
            y_true_all.append(y_batch)
            y_pred_batch = self.model.predict_on_batch(X_batch)
            y_pred_all.append(y_pred_batch)

        
        y_true_all = tf.concat(y_true_all,axis=0)
        y_pred_all = tf.concat(y_pred_all,axis=0)
        X_ = tf.concat(X_,axis=0)

        ## Ensure tensors are of type float32
        y_true_all = tf.cast(y_true_all,tf.float32)
        y_pred_all = tf.cast(y_pred_all,tf.float32)
        mse_per_observation = tf.reduce_mean(tf.square(y_true_all-y_pred_all),axis=1)


        return mse_per_observation.numpy()
    
    def on_epoch_end(self,epoch,logs=None):
        mse_train,X = self.compute_mse(self.train_dataset,epoch)
        mse_val = self.compute_mse(self.val_dataset)

        np.savetxt(r"C:\Users\SESA626862\Documents\RL_codes\outputs_mse\mse_train_" + str(epoch) + ".csv", mse_train, delimiter=",")

        self.mse_per_epoch_train.append(mse_train)
        self.mse_per_epoch_val.append(mse_val)

        ## print summary information
        mean_mse_train = np.mean(mse_train)
        mean_mse_val = np.mean(mse_val)
        print(f'Epoch {epoch + 1}: Mean MSE (Train): {mean_mse_train:.4f}, Mean MSE (val): {mean_mse_val:.4f}')
        """
        ## Plot histograms for training and validation MSE
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.hist(mse_train,bins=30,edgecolor='black')
        plt.yscale('log')
        plt.xlabel('MSE per observation (Train)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of MSE for Train Epoch {epoch + 1}')

        plt.subplot(1,2,2)
        plt.hist(mse_val,bins=30,edgecolor='black')
        plt.yscale('log')
        plt.xlabel('MSE per observation (Val)')
        plt.ylabel("Frequency")
        plt.title(f'Histogram of MSE for val epoch {epoch + 1}')

        plt.tight_layout()
        file_path = os.path.join(self.folder_path, f'histograms_epoch_{epoch + 1}.png')
        plt.savefig(file_path)
        plt.close()
        """
        
    def get_mse_per_epoch(self):
        return self.mse_per_epoch_train, self.mse_per_epoch_val

##data
A = pd.read_csv(r"C:\Users\SESA626862\Downloads\Alexander_upto_17.csv")
J = pd.read_csv(r"C:\Users\SESA626862\Downloads\Jones_upto_15_MIRRORS.csv")
H = pd.read_csv(r"C:\Users\SESA626862\Downloads\hompFly.csv")



############# Alexander and Jones
jones = J[~J['knot_id'].str.contains('!')]
jones = jones.drop(columns=['knot_id', 'representation','is_alternating','signature','minimum_exponent','maximum_exponent'])
jones['knot'] = jones['number_of_crossings'].astype(str) + '_' + jones['table_number'].astype(str)
jones = jones.drop(columns=['number_of_crossings','table_number'])
col = jones.pop('knot')
jones.insert(0, 'knot', col)
jones = jones.drop(columns=["knot"])

alexander = A[A["number_of_crossings"] < 16]
alexander['knot'] = alexander['number_of_crossings'].astype(str) + '_' + alexander['table_number'].astype(str)
col = alexander.pop('knot')
alexander.insert(0, 'knot', col)
alexander = alexander.drop(columns=["N/A_1","number_of_crossings","table_number","table_number","is_alternating","signature","minimum_exponent","maximum_exponent"])
alexander = alexander.drop(columns=["knot"])


###### Split the data
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(alexander,jones,H,test_size=0.2,random_state=42)
X1_test, X1_val, X2_test, X2_val, y_test, y_val = train_test_split(X1_test,X2_test,y_test,test_size=0.5,random_state=42)


X1_train_ = np.copy(X1_train)
X2_train_ = np.copy(X2_train)

pd.DataFrame(X1_train_).to_csv(r"C:\Users\SESA626862\Documents\RL_codes\outputs_mse\X1_train.csv")
pd.DataFrame(X2_train_).to_csv(r"C:\Users\SESA626862\Documents\RL_codes\outputs_mse\X2_train.csv")

X1_train,scale_alex = scale_data(X1_train)
X2_train,scale_jones = scale_data(X2_train)

X1_test,_ = scale_data(X1_test,scaler=scale_alex)
X2_test,_ = scale_data(X2_test,scaler=scale_jones)
X1_val,_ = scale_data(X1_val,scaler=scale_alex)
X2_val,_ = scale_data(X2_val,scaler=scale_jones)                                                                                                                                                                                                                     


## Reshape inputs to match the expected shape for LSTMs
X1_train = X1_train.reshape(-1,17,1)
X1_test = X1_test.reshape(-1,17,1)
X1_val = X1_val.reshape(-1,17,1)
X2_train = X2_train.reshape(-1,51,1)
X2_test = X2_test.reshape(-1,51,1)
X2_val = X2_val.reshape(-1,51,1)
y_train = y_train.values
y_test = y_test.values
y_val = y_val.values


## Create  Tensorflow datasets
def create_tf_dataset(X1,X2,y,batch_size = 32,buffer_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(((X1,X2),y))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset=create_tf_dataset(X1_train,X2_train,y_train)
test_dataset = create_tf_dataset(X1_test,X2_test,y_test)
val_dataset = create_tf_dataset(X1_val,X2_val,y_val)


###### Defining the model architecture

## LSTM for alexander
input1 = Input(shape=(17,1))
lstm1 = Bidirectional(LSTM(128,return_sequences = True))(input1)
attention1 = Attention()([lstm1,lstm1])
flattened1 = Flatten()(attention1)
dense1= Dense(128,activation='relu')(flattened1)
dropout1 = Dropout(0.2)(dense1)

## LSTM for jones
input2 = Input(shape=(51,1))
lstm2 = Bidirectional(LSTM(128,return_sequences = True))(input2)
attention2 = Attention()([lstm2,lstm2])
flattened2 = Flatten()(attention2)
dense2 = Dense(128,activation='relu')(flattened2)
dropout2 = Dropout(0.2)(dense2)

## Concatenate features
concatenated= Concatenate()([dropout1,dropout2])
dense = Dense(128,activation='relu')(concatenated)
output = Dense(152,activation='linear')(dense)

## The model
model = Model(inputs = [input1,input2], outputs=output)


####### Compile and train the model 
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error')

## Early stopping 
early_stopping = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

## Initialize the custom callback with the training dataset
per_observation_mse_callback = PerObservationMSECallback(train_dataset, val_dataset)


## Train the model
history = model.fit(train_dataset,epochs=100,batch_size=32,validation_data=val_dataset,callbacks=[early_stopping,per_observation_mse_callback], verbose=1)


###### Evaluate the model
y_pred = model.predict([X1_test,X2_test])
mse = mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test,y_pred)
print(f'r2 score: {r2}')

### Plot training and validation loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss LSTM 128')
plt.show()

## Plot predictions vs actuals values
plt.figure(figsize=(10,6))
plt.plot(y_test[0],label='Actual')
plt.plot(y_pred[0],label='Predicted')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions vs actuals for sample 1')
plt.show()

# Plot residuals
residuals = y_test-y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred,residuals,alpha=0.5)
plt.axhline(0,color="red",linestyle="--")
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()


