# Importations des bibliothèques nécessaires
import os
import tensorflow as tf
import mlflow.tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# Configuration des paramètres pour réduire les messages de log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Limite les logs à WARNING et ERROR uniquement
os.environ['TF_FORCE_CPU'] = 'true'  # Force l’utilisation du CPU uniquement

# Configurer l'URI de suivi pour utiliser un serveur MLflow distant
mlflow.set_tracking_uri("http://ec2-54-144-47-93.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("tagging_experiment_Vit")

# Fonction pour créer un générateur d'images avec augmentation
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

# Fonction pour créer les générateurs de données pour l'entraînement et la validation
def create_data_generators(img_generator, data_dir, target_size=(224, 224), batch_size=16):
    train_gen = img_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    val_gen = img_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    return train_gen, val_gen

# Initialiser le générateur d'images et les chemins de données
img_generator = create_image_generator()
data_dir = '../data'  # Chemin des données
train_gen, val_gen = create_data_generators(img_generator, data_dir)

# Fonction pour construire le modèle ViT avec transfert d'apprentissage
def build_vit_transfer_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    vit_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b16_fe/1", trainable=True)  # Chargement du modèle pré-entraîné
    x = vit_layer(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# Création et compilation du modèle
model_vit = build_vit_transfer_model(input_shape=(224, 224, 3), num_classes=5)
model_vit.compile(optimizer=AdamW(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec affichage de l'avancement
history_vit = model_vit.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    verbose=1  # Affichage de l'avancement de l'entraînement
)

# Inférer la signature pour le modèle à sauvegarder dans MLflow
signature = mlflow.models.signature.infer_signature(train_gen[0][0], model_vit.predict(train_gen[0][0]))

# Sauvegarder le modèle dans MLflow avec signature et enregistrer les paramètres et métriques
with mlflow.start_run(run_name="ViT_Best_Model"):
    mlflow.tensorflow.log_model(model_vit, "ViT_Best_Model", signature=signature)
    mlflow.log_params({
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "epochs": 10
    })
    mlflow.log_metrics({
        "train_accuracy": max(history_vit.history['accuracy']),
        "val_accuracy": max(history_vit.history['val_accuracy'])
    })
