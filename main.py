import pandas as pd
import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.models import model_from_json
from keras.layers import Embedding, Flatten, Dense

import joblib

import jieba
import langid
from translate import Translator

from google.cloud import storage
import io,os


# class Model:
#     def __init__(self, model_path, **args):
#
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./studio-433807-2a863d151232.json"
#         bucket_name = "xrspace_ex_bucket"
#         blob_name = 'emotion_detection_4_emotion/four_emotion_tokenizer.joblib'
#         blob_name_2 = 'emotion_detection_4_emotion/training_translate_4E.txt'
#         blob_name_3 = 'emotion_detection_4_emotion/four_emotion_LabelEncoder.model'
#         blob_name_4 = 'emotion_detection_4_emotion/model_architecture_four_emotion.json'
#         blob_name_5 = 'emotion_detection_4_emotion/model_weights_four_emotion.h5'
#
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(blob_name)
#         blob_txt = bucket.blob(blob_name_2)
#         blob_LabelEncoder = bucket.blob(blob_name_3)
#         blob_architecture = bucket.blob(blob_name_4)
#         blob_weights = bucket.blob(blob_name_5)
#
#         #################################################################
#         # four_emotion_tokenizer.joblib
#         self.model_tokenizer = io.BytesIO()
#         blob.download_to_file(self.model_tokenizer)
#         self.model_tokenizer.seek(0)
#
#         # training_translate_4E.txt
#         self.model_txt = io.BytesIO()
#         blob_txt.download_to_file(self.model_txt)
#         self.model_txt.seek(0)
#
#         # four_emotion_LabelEncoder.model
#         self.model_LabelEncoder = io.BytesIO()
#         blob_LabelEncoder.download_to_file(self.model_LabelEncoder)
#         self.model_LabelEncoder.seek(0)
#
#         # model_architecture_four_emotion.json
#         self.model_architecture = io.BytesIO()
#         blob_architecture.download_to_file(self.model_architecture)
#         self.model_architecture.seek(0)
#
#         # model_weights_four_emotion.h5
#         # self.model_weights = io.BytesIO()
#         # blob_weights.download_to_file(self.model_weights)
#         blob_weights.download_to_filename("C:/Users/XRSPACE/GCS/model_weights_four_emotion.h5")
#         # self.model_weights.seek(0)
#         #################################################################
#
#         # something about load model, e.g. model path, parameter, and more.
#         self.model_path = model_path
#         # data = pd.read_csv("training_translate_4E.txt", sep=',;,')
#         data = pd.read_csv(self.model_txt, sep=',;,')
#         data.columns = ["Text", "Emotions"]
#         texts = data["Text"].tolist()
#         for x in range(len(texts)):
#             texts[x] = ' '.join(jieba.cut(texts[x], cut_all=False, HMM=True))
#
#
#         # self.tokenizer = joblib.load('four_emotion_tokenizer.joblib')
#         self.tokenizer = joblib.load(self.model_tokenizer)
#
#
#         self.labels = data["Emotions"].tolist()
#         # self.label_encoder = joblib.load('four_emotion_LabelEncoder.model')
#         self.label_encoder = joblib.load(self.model_LabelEncoder)
#         self.max_length = 100
#         # sequences = self.tokenizer.texts_to_sequences(texts)
#         # self.max_length = max([len(seq) for seq in sequences])
#
#         langid.set_languages(['en', 'zh'])
#         self.translator = Translator(from_lang="en", to_lang="zh-tw")
#
#         json_string = blob_architecture.download_as_text()
#
#         # with open("model_architecture_four_emotion.json", "r") as text_file:
#         #     json_string = text_file.read()
#         #     self.model = model_from_json(json_string)
#         #     self.model.load_weights("model_weights_four_emotion.h5", by_name=False)
#         self.model = model_from_json(json_string)
#
#         # with h5py.File(self.model_weights, 'r') as f:
#         #     self.model.load_weights(f, by_name=False)
#
#         self.model.load_weights("C:/Users/XRSPACE/GCS/model_weights_four_emotion.h5", by_name=False)
#         # self.model.load_weights("model_weights_four_emotion.h5", by_name=False)
#
#         user_input = input("You: ")
#
#
#     def predict(self, user_input):
#         if langid.classify(user_input)[0] == "en":
#             translation = self.translator.translate(user_input)
#             user_input = translation
#         else:
#             user_input = user_input
#
#         input_text = ' '.join(jieba.cut(user_input, cut_all=False, HMM=True))
#
#         input_sequence = self.tokenizer.texts_to_sequences([input_text])
#
#         padded_input_sequence = pad_sequences(input_sequence, maxlen=self.max_length)
#
#         prediction = self.model.predict(padded_input_sequence)
#
#         if prediction[0][3] > 0.3:
#             predicted_label = ['neutral']
#         else:
#             predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction[0])])
#
#         # predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction[0])])
#         return predicted_label
#
#
# if __name__ == '__main__':
#     # 這裡是關鍵讓 Data team 知道怎麼和 model 溝通
#     path = "./"
#
#     model_Sequential = Model(path)
#
#     user_input = "生活真是太有趣了！"
#
#     output = model_Sequential.predict(user_input)
#     print(output)
#
#     assert output





os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./studio-433807-2a863d151232.json"
bucket_name = "xrspace_ex_bucket"

blob_name_1 = 'emotion_detection_4_emotion/four_emotion_tokenizer.joblib'
blob_name_2 = 'emotion_detection_4_emotion/training_translate_4E.txt'
blob_name_3 = 'emotion_detection_4_emotion/four_emotion_LabelEncoder.model'
blob_name_4 = 'emotion_detection_4_emotion/model_architecture_four_emotion.json'
blob_name_5 = 'emotion_detection_4_emotion/model_weights_four_emotion.weights.h5'

client = storage.Client()
bucket = client.bucket(bucket_name)

blob_tokenizer = bucket.blob(blob_name_1)
blob_txt = bucket.blob(blob_name_2)
blob_LabelEncoder = bucket.blob(blob_name_3)
blob_architecture = bucket.blob(blob_name_4)
blob_weights = bucket.blob(blob_name_5)

#################################################################
# four_emotion_tokenizer.joblib
model_tokenizer = io.BytesIO()
blob_tokenizer.download_to_file(model_tokenizer)
model_tokenizer.seek(0)

# training_translate_4E.txt
model_txt = io.BytesIO()
blob_txt.download_to_file(model_txt)
model_txt.seek(0)

# four_emotion_LabelEncoder.model
model_LabelEncoder = io.BytesIO()
blob_LabelEncoder.download_to_file(model_LabelEncoder)
model_LabelEncoder.seek(0)

# model_architecture_four_emotion.json
json_string = blob_architecture.download_as_text()

# model_weights_four_emotion.h5
# model_weights = io.BytesIO()
# blob_weights.download_to_file(model_weights)
# model_weights.seek(0)
blob_weights.download_to_filename("./model_weights_four_emotion.weights.h5")
#################################################################

data = pd.read_csv('training_translate_4E_4.txt', sep=',;,',engine='python')
data.columns = ["Text", "Emotions"]
texts = data["Text"].fillna("")
texts = texts.tolist()
labels = data["Emotions"].tolist()
for x in range(len(texts)):
            texts[x] = ' '.join(jieba.cut(texts[x], cut_all=False, HMM=True))

################################################################################
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# joblib.dump(tokenizer, 'four_emotion_tokenizer.joblib')
#
# max_length = 100
#
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(labels)
# joblib.dump(label_encoder,"four_emotion_LabelEncoder.model")
#
# one_hot_labels = keras.utils.to_categorical(labels)
#
# sequences = tokenizer.texts_to_sequences(texts)
# padded_sequences = pad_sequences(sequences, maxlen=max_length)
#
# xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences,
#                                                 one_hot_labels,
#                                                 test_size=0.2)
#
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
#                     output_dim=128, input_length=max_length))
# model.add(Flatten())
# model.add(Dense(units=128, activation="relu"))
# model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(xtrain, ytrain, epochs=5, batch_size=32, validation_data=(xtest, ytest))
#
#
#
# model_architecture = model.to_json()
# with open("model_architecture_four_emotion.json", "w") as json_file:
#     json_file.write(model_architecture)
#
# model.save_weights("model_weights_four_emotion.weights.h5")
# a= input("訓練完工")


################################################################################

user_input = "生活真是太有趣了！"
# tokenizer = joblib.load('four_emotion_tokenizer.joblib')
tokenizer = joblib.load(model_tokenizer)

labels = data["Emotions"].tolist()
label_encoder = joblib.load(model_LabelEncoder)
max_length = 100
langid.set_languages(['en', 'zh'])
translator = Translator(from_lang="en", to_lang="zh-tw")
# with open("model_architecture_four_emotion.json", "r") as text_file:
#     json_string = text_file.read()
#     model = Sequential()
#     model = model_from_json(json_string)
model = model_from_json(json_string)
# model.load_weights("C:/Users/XRSPACE/GCS/model_weights_four_emotion.h5")
model.load_weights("model_weights_four_emotion.weights.h5")

input_text = ' '.join(jieba.cut(user_input, cut_all=False, HMM=True))
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
print("自信 : ",'%.7f' % prediction[0][0])
print("興奮 : ",'%.7f' % prediction[0][1])
print("愉悅 : ",'%.7f' % prediction[0][2])
print("平淡 : ",'%.7f' % prediction[0][3])

if prediction[0][3] > 0.3:
    predicted_label = ['neutral']
else:
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])

print(predicted_label)
print(type(str(predicted_label)))


