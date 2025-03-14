import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Activation, Dense, Permute, Dropout, concatenate, add, dot, LSTM
import matplotlib.pyplot as plt

with open("train_qa.txt", "rb") as fp:
    train_data = pickle.load(fp)
with open("test_qa.txt", "rb") as fp:
    test_data = pickle.load(fp)

vocab = set()
all_data = train_data + test_data
for story, ques, ans in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(ques))

vocab.add('yes')
vocab.add('no')

vocab_len = len(vocab) + 1
max_story_len = max([len(data[0]) for data in all_data])
max_ques_len = max([len(data[1]) for data in all_data])

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

def vectorize_data(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_ques_len=max_ques_len):
    X = []
    Xq = []
    A = []
    for story, ques, ans in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in ques]
        a = np.zeros(len(word_index) + 1)
        a[word_index[ans]] = 1
        X.append(x)
        Xq.append(xq)
        A.append(a)
    return pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_ques_len), np.array(A)

input_train, queries_train, answers_train = vectorize_data(train_data)
input_test, queries_test, answers_test = vectorize_data(test_data)

input_sequence = Input((max_story_len,))
ques_sequence = Input((max_ques_len,))

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len, output_dim=64))
input_encoder_m.add(Dropout(0.2))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len, output_dim=max_ques_len))
input_encoder_c.add(Dropout(0.2))

ques_encoder = Sequential()
ques_encoder.add(Embedding(input_dim=vocab_len, input_length=max_ques_len, output_dim=64))
ques_encoder.add(Dropout(0.2))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = ques_encoder(ques_sequence)

match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2, 1))(response)

answer = concatenate([response, question_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, ques_sequence], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit([input_train, queries_train], answers_train,
                    validation_data=([input_test, queries_test], answers_test),
                    batch_size=30,
                    epochs=15)

loss, accuracy = model.evaluate([input_test, queries_test], answers_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save("chatbot_model.h5")

def predict_answer(input_text, question_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_story_len)
    ques_seq = tokenizer.texts_to_sequences([question_text])
    ques_seq = pad_sequences(ques_seq, maxlen=max_ques_len)
    
    prediction = model.predict([input_seq, ques_seq])
    predicted_word_index = np.argmax(prediction)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

input_text = input()
question_text = input()
print("Predicted answer:", predict_answer(input_text, question_text))
