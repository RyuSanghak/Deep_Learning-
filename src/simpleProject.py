import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras




data = pd.read_csv('gpascore.csv')
print(data)

#print(data.isnull().sum())
data = data.dropna() # erase row which is empty

yData = data['admit'].values

xData = []

for i, rows in data.iterrows():
    xData.append([rows['gre'], rows['gpa'], rows['rank'] ])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(256, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'), # get result between 0 and 1 
    ])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) , loss='binary_crossentropy' , metrics=['accuracy'])

model.fit( np.array(xData), np.array(yData), epochs=1000) # x = learning data, y = answer 


# Prediction 
predictValue = model.predict(np.array([[750, 3.7, 3], [400, 2.2, 1]]))
print(predictValue)

model.save('saved_model/my_model.keras') # save model 

#[ [380, 3.21, 3], [660, 3.67, 3], [], []]
#[ 0, 1, 1, 0, 1,]
exit()

model = tf.keras.models.load_model('saved_model/my_model.keras')

predictValue = model.predict(np.array([[100, 3.7, 3], [900, 4.0, 1]]))
print(predictValue)



exit()
