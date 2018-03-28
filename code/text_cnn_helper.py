from collections import Counter
import itertools
from sklearn import preprocessing
import numpy as np
import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model

def pad_sentences(x_train,x_test, padding_word="<PAD/>"):

    sequence_length = max(len(x) for x in x_train)
    
    padded_x_train = []
    for i in range(len(x_train)):
        sentence = x_train[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_x_train.append(new_sentence)
      
    padded_x_test = []
    for i in range(len(x_test)):
        sentence = x_test[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_x_test.append(new_sentence)
        
    return padded_x_train,padded_x_test

def build_vocab(sentences):

    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    
    return [vocabulary, vocabulary_inv]

def build_input_data(x_train, x_test, vocabulary):

    x_train = np.array([[vocabulary[word] for word in sentence] for sentence in x_train])
    x_test = np.array([[vocabulary[word] for word in sentence] for sentence in x_test])
    
    return [x_train, x_test]

def model_load(x_train, vocabulary_inv):
    
    sequence_length = x_train.shape[1] 
    vocabulary_size = len(vocabulary_inv) 
    embedding_dim = 256
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5

    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=20, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
