from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
import matplotlib.pyplot as plt


def create_model(input_dim, category_num):
    model = Sequential()
    model.add(Convolution2D(32,3,1,input_shape=(input_dim),padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 

    model.add(Convolution2D(64,3,1,padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,1,padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,1,padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
 
    model.add(Dense(category_num))
    model.add(Activation('softmax'))
 
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Draw a diagram of Accracy and loss.
def create_result_map(history):
    train_accuracy = history.history['accuracy']
    train_loss = history.history['loss']
    validation_accuracy = history.history['val_accuracy']
    validation_loss = history.history['val_loss']
    epochs = range(1,len(train_accuracy)+1)
    plt.plot()
    plt.plot(epochs,train_accuracy,label='training_accuracy')
    plt.plot(epochs,validation_accuracy,label='validation_accuracy')
    plt.title('training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.figure()

    plt.plot(epochs,train_loss,label='training_loss')
    plt.plot(epochs,validation_loss,label='validation_loss')
    plt.title('training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Normalization process.
def preprocessing(data):
    data = data.astype('float32')
    data /= 255
    return data

if __name__ == '__main__':
    # Download CIFAR-10 image
    # (nb_samples, nb_rows, nb_cols, nb_channel) = tf
    (x_train, y_train),(x_test,y_test) = cifar10.load_data()
    input_dim = x_train.shape[1:] 
    category_num = 10
    batch_size = 128
    epoch = 50

    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)

    y_train = np_utils.to_categorical(y_train, category_num)
    y_test = np_utils.to_categorical(y_test, category_num)

    model = create_model(input_dim,category_num)
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=True
    )
    create_result_map(history)

    # Evaluation of the trained model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('# Test')
    print('loss : ', test_loss)
    print('accuracy : ', test_accuracy)