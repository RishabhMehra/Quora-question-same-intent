
import numpy as np
import pickle
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, LSTM, Embedding, Concatenate, CuDNNLSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.15

############################### LOAD AND FORMAT DATA ########################################
with open('dataset.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

question1 = []
question2 = []
labels = []
for i in dataset[:20000]:
    question1.append(i['question1'])
    question2.append(i['question2'])
    labels.append(i['is_duplicate'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(question1+question2)
question1_sequences = tokenizer.texts_to_sequences(question1)
question2_sequences = tokenizer.texts_to_sequences(question2)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

question1_data = pad_sequences(question1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
question2_data = pad_sequences(question2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.asarray(labels)
print('Shape of Question 1 data tensor:', question1_data.shape)
print('Shape of Question 2 data tensor:', question2_data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(question1_data.shape[0])
np.random.shuffle(indices)
question1_data = question1_data[indices]
question2_data = question2_data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * question1_data.shape[0])

q1_train = question1_data[:-nb_validation_samples]
q1_val = question1_data[-nb_validation_samples:]
q2_train = question2_data[:-nb_validation_samples]
q2_val = question2_data[-nb_validation_samples:]
y_train = labels[:-nb_validation_samples]
y_val = labels[-nb_validation_samples:]

########################### CREATE EMBEDDING MATRIX AND EMBEDDING LAYER ########################
with open('embeddings_index.pkl', 'rb') as handle:
    embeddings_index  = pickle.load(handle)
    
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


############################# MODEL 1 #########################################################

input_q1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
input_q2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences_q1 = embedding_layer(input_q1)
embedded_sequences_q2 = embedding_layer(input_q2)

q1_lstm_layer_1 = CuDNNLSTM(MAX_SEQUENCE_LENGTH)(embedded_sequences_q1)
q2_lstm_layer_1 = CuDNNLSTM(MAX_SEQUENCE_LENGTH)(embedded_sequences_q2)

merge = Concatenate()([q1_lstm_layer_1,q2_lstm_layer_1])

dense_1 = Dense(256, activation='relu')(merge)
dense_2 = Dense(256,activation='relu')(dense_1)
output = Dense(1,activation='sigmoid')(dense_2)

model = Model([input_q1,input_q2], output)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit([q1_train, q2_train], y_train,
            batch_size=128, nb_epoch=10,
            validation_data=([q1_val, q2_val], y_val))

############################# MODEL 2 #########################################################

input_q1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
input_q2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences_q1 = embedding_layer(input_q1)
embedded_sequences_q2 = embedding_layer(input_q2)

q1_lstm_layer_1 = CuDNNLSTM(MAX_SEQUENCE_LENGTH, return_sequences=True)(embedded_sequences_q1)
q2_lstm_layer_1 = CuDNNLSTM(MAX_SEQUENCE_LENGTH, return_sequences=True)(embedded_sequences_q2)

dropout_q1 = Dropout(0.2)(q1_lstm_layer_1)
dropout_q2 = Dropout(0.2)(q2_lstm_layer_1)

q1_lstm_layer_2 = CuDNNLSTM(MAX_SEQUENCE_LENGTH)(dropout_q1)
q2_lstm_layer_2 = CuDNNLSTM(MAX_SEQUENCE_LENGTH)(dropout_q2)

merge = Concatenate()([q1_lstm_layer_2,q2_lstm_layer_2])

dense_1 = Dense(256, activation='relu')(merge)
output = Dense(1,activation='sigmoid')(dense_1)

model = Model([input_q1,input_q2], output)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit([q1_train, q2_train], y_train,
            batch_size=128, nb_epoch=10,
            validation_data=([q1_val, q2_val], y_val))
