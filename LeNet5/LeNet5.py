import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# set up GPU as the first GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LeNet5(Model):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5, 5), padding = 'same', activation = 'relu')
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(16, kernel_size=(5,5), activation = 'relu')
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# model = Sequential()
# model.add(Conv2D(6, kernel_size=(5, 5), padding = 'same', activation = 'relu', input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, kernel_size=(5,5), activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = LeNet5(num_classes=10)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', \
    metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train[...,tf.newaxis], x_test[...,tf.newaxis]
model.fit(x_train, y_train, batch_size=24, epochs=80,\
     validation_data=(x_test, y_test), callbacks = callbacks)

