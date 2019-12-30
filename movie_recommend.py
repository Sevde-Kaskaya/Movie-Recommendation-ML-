import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.models import load_model

dataset = pd.read_csv('ratings.csv')
print (dataset.head())
print (dataset.shape)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.user_id.unique())
n_movies = len(dataset.movie_id.unique())

print ("\n\n\n", "number of users: ", n_users, "number of movies: ", n_movies)

# movie vektorunu olusturuyoruz
movie_input = Input(shape=[1], name="Movie-Input")
movie_embedding = Embedding(n_movies+1, 5, name="Movie-Embedding")(movie_input)
movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

# kullanici vektorunu olusturuyoruz
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# featuresları baglıyoruz
conc = Concatenate()([movie_vec, user_vec])

# cikis katmanı ekliyoruz
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# modelin training oncesi config
model2 = Model([user_input, movie_input], out)
model2.compile('adam', 'mean_squared_error')

if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.user_id, train.movie_id], train.rating, epochs=5, verbose=1)
    model2.save('regression_model2.h5')
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")

model2.evaluate([test.user_id, test.movie_id], test.rating)
predictions = model2.predict([test.user_id.head(10), test.movie_id.head(10)])

for i in range(0,10):
	print(predictions[i], test.rating.iloc[i])