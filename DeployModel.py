from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, Bidirectional, RepeatVector, Flatten
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DeployModel:
    def __init__(self, df, units=100, batch_size=30, epochs=1500, optimizer='adam', loss='mae', dropout=0.1, window_length=12):
        if not isinstance(df, pd.DataFrame) or df.shape[0] <= window_length:
            raise ValueError("df must be a DataFrame with enough rows to create the dataset.")

        self.nb_label_feature = 7
        self.UNITS = units
        self.BATCHSIZE = batch_size
        self.EPOCH = epochs
        self.OPTIMIZER = optimizer
        self.LOSS = loss
        self.DROPOUT = dropout
        self.window_length = window_length
        self.number_of_features = df.shape[1]
        self.es = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=100)

        self.model = None

    def define_model(self, number_of_features, nb_label_feature):
        model = Sequential()
        model.add(Input(shape=(self.window_length, number_of_features)))
        model.add(LSTM(self.UNITS, return_sequences=False))
        model.add(Dense(nb_label_feature))
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=['acc'])
        return model

    def define_bidirectional_model(self, number_of_features, nb_label_feature):
        model = Sequential()
        model.add(Input(shape=(self.window_length, number_of_features)))
        model.add(Bidirectional(LSTM(self.UNITS, dropout=self.DROPOUT, return_sequences=False)))
        model.add(Dense(nb_label_feature))
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=['acc'])
        return model

    def define_autoencoder_model(self, number_of_features, nb_label_feature):
        model = Sequential()
        model.add(Input(shape=(self.window_length, number_of_features), return_sequences=True))
        model.add(LSTM(self.UNITS, return_sequences=False))
        model.add(RepeatVector(self.window_length))
        model.add(LSTM(self.UNITS, dropout=self.DROPOUT, return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(TimeDistributed(Dense(number_of_features)))
        model.add(Flatten())
        model.add(Dense(nb_label_feature))
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=['acc'])
        return model

    def create_lstm_dataset(self, df):
        number_of_rows = df.shape[0]
        number_of_features = df.shape[1]
        scaler = StandardScaler().fit(df.values)
        transformed_dataset = scaler.transform(df.values)
        transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

        train = np.empty([number_of_rows - self.window_length, self.window_length, number_of_features], dtype=float)
        label = np.empty([number_of_rows - self.window_length, self.nb_label_feature], dtype=float)

        for i in range(0, number_of_rows - self.window_length):
            train[i] = transformed_df.iloc[i:i + self.window_length, 0:number_of_features]
            label[i] = transformed_df.iloc[i + self.window_length, 0:self.nb_label_feature]

        return train, label, scaler

    def train_model(self, train, labels, validation_data=None, model_type='simple'):
        if model_type == 'simple':
            self.model = self.define_model(self.number_of_features, self.nb_label_feature)
        elif model_type == 'bidirectional':
            self.model = self.define_bidirectional_model(self.number_of_features, self.nb_label_feature)
        elif model_type == 'autoencoder':
            self.model = self.define_autoencoder_model(self.number_of_features, self.nb_label_feature)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        callbacks = [self.es]

        if validation_data:
            validation_data = (validation_data[0], validation_data[1])

        history = self.model.fit(
            train,
            labels,
            batch_size=self.BATCHSIZE,
            epochs=self.EPOCH,
            validation_data=validation_data,
            callbacks=callbacks
        )
        return history

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")

        predictions = self.model.predict(input_data)

        scaled_predicted_output = np.pad(predictions, ((0, 0), (0, 13)), 'constant')
        predicted_numbers = self.scaler.inverse_transform(scaled_predicted_output)

        predicted_boules = np.clip(predicted_numbers[:, :5], 1, 50)
        predicted_etoiles = np.clip(predicted_numbers[:, 5:], 1, 12)

        predicted_numbers = np.concatenate([predicted_boules, predicted_etoiles], axis=1)
        predicted_numbers = predicted_numbers.astype(int)

        formatted_numbers = ', '.join(map(str, predicted_numbers[0]))
        print("Prédiction des nombres:", formatted_numbers)

        return predicted_numbers
