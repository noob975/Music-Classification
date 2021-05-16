# -*- coding: utf-8 -*-
"""
Created on Sat May  8 20:08:28 2021

@author: Aniruddha
"""

from python_speech_features import mfcc
import numpy as np
import os
import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import librosa

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

#%%
#Data Visualization

audio_path="C:\\Users\Aniruddha\Desktop\DL Stuff\genres\\blues\\blues.00000.wav"
x , sr = librosa.load(audio_path)


import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x,sr=sr)
n0 = 0000
n1 = 1000
        
plt.figure(figsize=(14, 5))

plt.plot(x[n0:n1]) #zoomed-in plot
plt.grid()

#Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
plt.colorbar()

#%%
#making separate folders to store 3-second clips of original 30 second songs.

genres = 'blues classical country disco pop hiphop jazz metal reggae rock'
genres = genres.split()
for g in genres:
  path_audio = os.path.join('C:\\Users\Aniruddha\Desktop\DL Stuff/content/audio3sec',f'{g}')
  os.makedirs(path_audio)
#%%
#Expanding the dataset by cutting up the 30 second files into 10x3-second sub audios.

#make x-second intervals of any song
def three(filepath, how_many_intervals, how_many_seconds, export_path, songname='splitted'):
    from pydub import AudioSegment

    for w in range(0,how_many_intervals):
        #print(i)
        t1 = how_many_seconds*(w)*1000
        t2 = how_many_seconds*(w+1)*1000
        newAudio = AudioSegment.from_wav(filepath)
        new = newAudio[t1:t2]
        new.export(f'{export_path}\\{str(songname)+" "+str(w)}.wav', format="wav")

#%%
for g in genres:
  print(f"{g}")
  for filename in os.listdir(os.path.join('C:\\Users\Aniruddha\Desktop\DL Stuff\genres',f"{g}")):
    song  =  os.path.join(f'C:\\Users\Aniruddha\Desktop\DL Stuff\genres\{g}',f'{filename}')
    three(filepath=song, how_many_intervals=10, how_many_seconds=3, export_path=f'C:\\Users\Aniruddha\Desktop\DL Stuff\content\\audio3sec\\{g}',songname=g)
#%%
""" 
Examples of Feature extraction
Only for reference. Do not run in actual code,
"""


# #Zero Crossings 
# zero_crossings = librosa.zero_crossings(x, pad=False)
# print("Number of zero crossings:", sum(zero_crossings))

# #Spectral Centroids: center of mass of sound. Weighted mean of frequencies.
# spectral_centroids = librosa.feature.spectral_centroid(x[n0:n1], sr=sr)[0]
# print("Spectral centroids shape:", spectral_centroids.shape)

# # Computing the time variable for visualization
# frames = range(len(spectral_centroids))
# t = librosa.frames_to_time(frames)

# # Normalising the spectral centroid for easier visualisation
# def normalize(x, axis=0):
#     return sklearn.preprocessing.minmax_scale(x, axis=axis)
# #Plotting the Spectral Centroid along the waveform
# librosa.display.waveplot(x[n0:n1], sr=sr, alpha=0.5)
# plt.plot(t, normalize(spectral_centroids), color='r')
# plt.title("Spectral Centroid")
# plt.show()

# #Spectral Rolloff: frequncy below which specfied % (usually 85) of energy lies
# spectral_rolloff = librosa.feature.spectral_rolloff(x[n0:n1], sr=sr)[0]
# librosa.display.waveplot(x[n0:n1], sr=sr, alpha=0.4)
# plt.title("Spectral Rolloff")
# plt.plot(t, normalize(spectral_rolloff), color='r')
# plt.show()

# MFCC: Describe the shape of spectral envelope. Coefficients obtained after DCT. By default 20. 
# mfcc = librosa.feature.mfcc(x, sr=sr)
#%%
#Calculating features of GTZAN dataset and storing in csv file. Takes a long time.

#creating header row for csv file
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#writing into the file 
file = open('3sec_data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
    print(g)
    for filename in os.listdir(os.path.join('C:\\Users\Aniruddha\Desktop\DL Stuff\content\\audio3sec',f"{g}")):
        songname = os.path.join(f'C:\\Users\Aniruddha\Desktop\DL Stuff\content\\audio3sec\{g}',f'{filename}')
        y, sr = librosa.load(songname, mono=True, duration=3)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('3sec_data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

#%%
#Preparing training, validation and test sets.

#removing "bluesxxxxx.wa"v type of data from csv
music=pd.read_csv("C:\\Users\Aniruddha\Desktop\DL Stuff\\3sec_data.csv")
music = music.drop(['filename'],axis=1) 

#numerically encoding music genres
genre_list = music.iloc[:, -1]
encoder = LabelEncoder()
encoded_genres = encoder.fit_transform(genre_list)

#normalizing data
scaler = MinMaxScaler()
normalized_X = scaler.fit_transform(np.array(music.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(normalized_X, encoded_genres, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#%%
#model making and training

model=keras.Sequential([
    layers.Dense(64,activation='relu',input_shape=[X_train.shape[1]]),
    # layers.Dropout(rate=0.2),
    layers.Dense(128,activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Dense(256,activation='relu'),
    # layers.Dropout(rate=0.2),
    layers.Dense(64,activation='relu'),
    # layers.Dropout(rate=0.2),    
    layers.Dense(10,activation='softmax')])

opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=80,
    verbose=0,
)


test_loss, test_acc = model.evaluate(X_test,y_test)
print('Using MLFFNN: ',test_acc)

train_loss, train_acc = model.evaluate(X_train,y_train)
print('Using MLFFNN (on training data): ',train_acc)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
#%%

#making confusion matrix

predictor=model.predict(X_test)
pred=[]

for i in predictor:
    pred.append(list(i).index(max(i)))

#decode numbered classes back to worded classes
def convert_to_text(numbers,texts):
    converted=[]
    for number in numbers:
        converted.append(texts[number])
    return converted

converted_pred=convert_to_text(pred, genres)
converted_y_test=convert_to_text(y_test, genres)

cm=confusion_matrix(converted_y_test, converted_pred, labels=genres)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + genres)
ax.set_yticklabels([''] + genres)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# filepath = "C:\\Users\Aniruddha\Desktop\DL Stuff\genres\\Always Somewhere.wav"

        
        


