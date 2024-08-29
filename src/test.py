import tensorflow as tf
import keras

tall = 170
shoeSize = 260

# shoeSize = tall * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_func():
    predictionValue = tall * a + b
    return tf.square(shoeSize - predictionValue)
    #return tf.square(realValue - predictionValue)
    
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    

    with tf.GradientTape() as tape:
        loss = loss_func()
        
    grads = tape.gradient(loss, [a,b])

    opt.apply_gradients(zip(grads, [a,b]))

    print(a.numpy(), b.numpy() )
    print(170*a.numpy()+b.numpy())
    
    
    
