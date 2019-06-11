from keras.layers import Input, Dense, RNN, LSTM, Concatenate, MaxPooling1D, Embedding
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import model_from_json
import numpy as np
import pandas as pd
from time import time

print("-----Start-----\n")

df1 = pd.read_hdf("jsons/yearwiseX1.h5",'main_dfX1')
df2 = pd.read_hdf("jsons/yearwiseX2.h5","main_dfX2")
df3 = pd.read_hdf("jsons/yearwisey.h5","main_dfy")

print("-----Data Loaded-----\n")

df1 = df1.apply(lambda x: np.array(x))
df2 = df2.apply(lambda x: np.array(x))
input1 = np.asarray(df1.tolist())
input2 = np.asarray(df2.tolist())
input1 = input1.reshape(input1.shape[0] , input1.shape[1] , 1)
input2 = input2.reshape(input2.shape[0] , input2.shape[1] , 1)
main_output = df3.values
main_output= main_output.reshape(main_output.shape[0],1,1)
batch_size = 512
epochs = 8

main_input_1 = Input(shape=(50,1), name='main_input_1')
main_input_2 = Input(shape=(50,1), name='main_input_2')
lstm_out=LSTM(32,activation='tanh',recurrent_activation='sigmoid',return_sequences=True)
max_pooling=MaxPooling1D(pool_size=2,strides=50,padding='valid')
dense = Dense(10000,activation='relu')
dense2 = Dense(10000,activation='relu')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

lstm_out_1=lstm_out(main_input_1)
lstm_out_2=lstm_out(main_input_2)
max_pooling_1=max_pooling(lstm_out_1)
max_pooling_2=max_pooling(lstm_out_2)


concatenate_layer=Concatenate()([max_pooling_1,max_pooling_2])
dense_output1 = dense2(concatenate_layer)

dense_output2 = dense(dense_output1)
logistic_regression_output=Dense(1,activation='sigmoid',name='main_output')(dense_output2)


model = Model(inputs=[main_input_1, main_input_2], outputs=[logistic_regression_output])

model.compile(optimizer='RMSprop',
              loss={'main_output': 'mean_squared_error'},
              metrics=['mse', 'mae', 'mape', 'cosine'])

tensorboard = TensorBoard(log_dir="logs/".format(time()))

checkpoint = ModelCheckpoint("checkpoint", monitor='val_acc', verbose=1, mode='max')

model.fit({'main_input_1': input1, 'main_input_2': input2},
          {'main_output': main_output},
          epochs=epochs, 
          verbose=1,
          batch_size=batch_size,
          callbacks=[tensorboard,checkpoint],
          validation_split=0.2)

print("-----Training Completed-----\n")


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk\n")

print("-----Complete-----\n")