from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam

def load_model(x_train):
        
    model = Sequential()
    model.add(Dense(540, init='glorot_uniform', activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(180, init='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model    