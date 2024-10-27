# Importations des bibliothèques TensorFlow et autres utilitaires
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, Activation, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.applications import Xception
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras import layers
from keras import initializers
import keras
keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

# Importations pour le traitement des données et affichages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os


from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array



### Prétraitement des données

# Fonction pour créer le générateur d'images avec augmentation
def create_image_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.25,
        height_shift_range=0.25,
        rescale=1./255,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=(0.9, 1.1),
        channel_shift_range=0.1,
        vertical_flip=True,
        validation_split=0.2
    )

# Création des générateurs d'entraînement et de validation
def create_data_generators(img_generator, data_dir, target_size=(224, 224), batch_size=16):
    train_generator = img_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    validation_generator = img_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    return train_generator, validation_generator


### Fonctions pour afficher le DataFrame et le graphique
def load_data_and_plot(path, list_of_breed):
    """
    Crée un DataFrame à partir des informations sur les races de chiens, affiche le DataFrame,
    calcule le nombre total d'images et trace un graphique du nombre d'images par race.
    
    Parameters:
    - path: Chemin vers le dossier contenant les sous-dossiers d'images par race.
    - list_of_breed: Liste des sous-dossiers (identifiant et nom de la race, formatés sous 'Id-name').
    
    """
    # Créer le DataFrame avec toutes les informations en une seule fois
    df = pd.DataFrame({
        'Id_name_breed': list_of_breed,
        'breed_name': [x.split('-')[1] for x in list_of_breed],
        'Id': [x.split('-')[0] for x in list_of_breed],
        'nb_pic': [len(os.listdir(os.path.join(path, name))) for name in list_of_breed]
    })

    # Afficher le DataFrame
    display(df)

    # Calculer et afficher le nombre total d'images
    total_images = df["nb_pic"].sum()
    print(f"\nLe nombre total des images est: {total_images}")

    # Créer le graphique du nombre d'images par race
    fig, ax = plt.subplots(figsize=(12,6))
    colors = ['red', 'green', 'blue', 'purple', 'orange']  
    df.plot(x='breed_name', y='nb_pic', kind="bar", ax=ax, legend=False, color=colors)
    
    # Personnaliser le titre du graphique
    ax.set_title("Images per Breed", fontsize=16)
    
    # Affichage optimisé du graphique
    plt.tight_layout()
    plt.show()

### Fonctions d'évaluation des modèles

# Affichage de la matrice de confusion
def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g')
    ax.set_xlabel("Étiquettes prédites")
    ax.set_ylabel("Vraies étiquettes")
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=45)
    plt.title("Matrice de Confusion")
    plt.show()

# Affichage du rapport de classification
def classification_report_df(y_true, y_pred_classes, class_names):
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(2)
    return df_report

### Affichage des courbes d'entraînement

# Courbes de perte
def plot_loss_curves(history, model_name="Model"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label=f"Training loss ({model_name})")
    plt.plot(history.history["val_loss"], label=f"Validation loss ({model_name})")
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Courbes d'accuracy
def plot_accuracy_curves(history, model_name="Model"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label=f"Training Accuracy ({model_name})")
    plt.plot(history.history["val_accuracy"], label=f"Validation Accuracy ({model_name})")
    plt.title(f'Training and Validation Accuracy - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

### Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

# Classe PatchEncoder
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

# Fonction MLP (perceptron multicouche)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Modèle complet ViT (Vision Transformer)
def build_vit_model(input_shape=(224, 224, 3), num_classes=5, patch_size=16, num_patches=196,
                    projection_dim=64, transformer_layers=8, num_heads=4, transformer_units=[256, 64], 
                    mlp_head_units=[128, 64]):

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

### Modèles de transfert learning avec finetuning : 

def create_xception_model(input_shape=(299, 299, 3), num_classes=5, dropout_rate=0.3, fine_tune_start=51):
    # Charger le modèle Xception pré-entraîné sur ImageNet sans les couches fully connected
    base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Geler les premières couches du modèle Xception jusqu'à `fine_tune_start`
    for layer in base_model_xception.layers[:fine_tune_start]:
        layer.trainable = False
    for layer in base_model_xception.layers[fine_tune_start:]:
        layer.trainable = True

    # Créer un modèle séquentiel
    model = Sequential([
        # Ajouter le modèle Xception pré-entraîné comme base
        base_model_xception,
        
        # Ajouter les nouvelles couches fully connected
        GlobalAveragePooling2D(),  # Pooling global pour réduire la dimensionnalité
        Dense(512),  # Couche Dense avec 512 unités
        BatchNormalization(),  # Normalisation pour stabiliser l'apprentissage
        Activation('relu'),  # Activation ReLU
        Dropout(dropout_rate),  # Dropout pour éviter le surapprentissage
        Dense(num_classes, activation='softmax')  # Couche de sortie pour la classification avec Softmax
    ])
    
    return model



# Modèle Vision Transformer (ViT)
def build_vit_transfer_model(input_shape=(224, 224, 3), num_classes=5):
    # Définir l'entrée du modèle avec la forme spécifiée
    inputs = Input(shape=input_shape)
    
    # URL du modèle Vision Transformer pré-entraîné sur TensorFlow Hub
    vit_model_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    
    # Charger le ViT comme une couche Keras, marqué comme entraînable pour permettre le fine-tuning
    vit_layer = hub.KerasLayer(vit_model_url, trainable=True)
    
    # Passer les données d'entrée à travers le modèle ViT
    x = vit_layer(inputs)
    
    # Ajouter une couche Dense pour l'apprentissage des caractéristiques de haut niveau
    # Activation ReLU est utilisée pour ajouter de la non-linéarité
    x = Dense(512, activation='relu')(x)
    
    # Appliquer le dropout pour réduire le surajustement lors du fine-tuning
    x = Dropout(0.5)(x)
    
    # Couche de sortie avec activation softmax pour la classification
    # Softmax est utilisé pour calculer la probabilité de chaque classe
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Créer et retourner le modèle Keras
    model = Model(inputs=inputs, outputs=outputs)
    return model



# Fonction pour afficher l'image et la prédiction avec un modèle donné
def predict_breed(image_path, model, input_shape=(224, 224)):
    # Charger l'image
    image = Image.open(image_path)
    
    # Afficher l'image 
    plt.imshow(image)
    plt.axis('on')  
    plt.show()

    # Redimensionner l'image à la taille attendue par le modèle
    image = image.resize(input_shape)

    # Prétraitement de l'image
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour correspondre à l'entrée du modèle
    img = img / 255.0  # Normaliser l'image

    # Prédiction avec le modèle donné
    prediction = model.predict(img)

    # Récupérer la classe prédite
    predicted_class = np.argmax(prediction)

    # Liste des noms de races
    breed_names = ['Japanese Spaniel', 'English Foxhound', 'Silky Terrier', 'Golden Retriever', 'German Shepherd']
    predicted_name = f"Nom prédit : {breed_names[predicted_class]}"

    # Afficher le nom prédit
    print(predicted_name)
