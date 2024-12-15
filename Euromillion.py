
import numpy as np
import matplotlib.pyplot as plt

from ImportData import ImportData
from TransformData import TransformData
from DeployModel import DeployModel

# Parameters
pathfolder = '/Users/soidi/Desktop/Projet/Python/euromillion/'
filenames = ['euromillions_202002-2.csv']
window_length = 12
nb_label_feature = 7

# Data Import
raw_data = ImportData(pathfolder, filenames)
df = raw_data.importFile()
df = raw_data.mergeFile()

# Data Transormation
df_transformed = TransformData.transform_data(df)

print(df.head(6))

# Model Initialisation
model = DeployModel(df)

# Dataset Creation
train, labels, scaler = model.create_lstm_dataset(df)

# Model Training
history = model.train_model(train, labels, model_type='simple')

# Print loss history
plt.plot(history.history['loss'])
plt.legend(['train_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Prediction on last X-attempts
last_twelve = df.tail(window_length)

# Scaler les données en utilisant le scaler approprié (en utilisant les mêmes colonnes que pour l'entraînement)
scaled_to_predict = scaler.transform(last_twelve.values)

# Effectuer la prédiction
scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))

# Assurez-vous que la forme est correcte avant de procéder
print(f"Shape of predicted output: {scaled_predicted_output_1.shape}")

# Si vous avez un scaler ajusté spécifiquement pour les prédictions, utilisez-le ici
# Inverse transform pour récupérer les valeurs d'origine pour les 7 premières colonnes (5 boules + 2 étoiles)
predicted_numbers = scaler.inverse_transform(np.concatenate([scaled_predicted_output_1, np.zeros((scaled_predicted_output_1.shape[0], df.shape[1] - nb_label_feature))], axis=1))

# Diviser en deux groupes : 1 à 5 (entre 0 et 50), 6 à 7 (entre 1 et 12)
predicted_numbers_1_to_5 = predicted_numbers[:, :5]
predicted_numbers_1_to_5 = np.clip(predicted_numbers_1_to_5, 0, 50)

predicted_numbers_6_to_7 = predicted_numbers[:, 5:]
predicted_numbers_6_to_7 = np.clip(predicted_numbers_6_to_7, 1, 12)

# Recombiner les prédictions
predicted_numbers = np.concatenate([predicted_numbers_1_to_5, predicted_numbers_6_to_7], axis=1)

# Conversion en entiers et affichage du résultat
predicted_numbers = predicted_numbers.astype(int)
formatted_numbers = ', '.join(map(str, predicted_numbers[0]))  # Utiliser la première ligne si le tableau est 2D
print("Prédiction des nombres:", formatted_numbers)
