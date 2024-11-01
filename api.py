import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import mlflow
import boto3
from botocore.config import Config 

# Configurer les identifiants AWS pour boto3
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

# Configurer boto3 avec un timeout plus long pour éviter les interruptions
config = Config(connect_timeout=60, read_timeout=60)
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=config
)

# Configurer MLflow pour utiliser un serveur de suivi distant
mlflow.set_tracking_uri("http://ec2-54-144-47-93.compute-1.amazonaws.com:5000/")

# Affichage du titre avec un style amélioré
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de la race de chiens avec ViT-B16</h1>", unsafe_allow_html=True)
st.write("Chargement des modèles et artefacts depuis MLflow...")

# Fonction pour charger un modèle MLflow avec mise en cache
@st.cache_resource
def load_mlflow_model(model_uri, model_type='tensorflow'):
    try:
        if model_type == 'tensorflow':
            return mlflow.tensorflow.load_model(model_uri)  
        else:
            st.error("Type de modèle non supporté")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_uri}: {str(e)}")
        return None

# Charger le modèle ViT-B16 en tant que modèle TensorFlow depuis MLflow
model_uri_vit_b16 = 'runs:/54c75fc613b5411b9710a2404c9fe70c/ViT_Best_Model'
model_vit_b16 = load_mlflow_model(model_uri_vit_b16, 'tensorflow')

# Téléchargement d'image par l'utilisateur
uploaded_file = st.file_uploader("Choisir une image de chien...", type="jpg")

if uploaded_file is not None:
    # Afficher l'image téléchargée et la centrer
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True)

    # Bouton pour déclencher la prédiction
    if st.button('Prédire'):
        if model_vit_b16:
            # Redimensionner et prétraiter l'image
            image = image.resize((224, 224))
            img = img_to_array(image)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Prédiction avec le modèle ViT-B16
            prediction_vit_b16 = model_vit_b16.predict(img)
            predicted_class_vit_b16 = np.argmax(prediction_vit_b16)

            # Liste des noms de races
            breed_names = ['Japanese_spaniel', 'English_foxhound', 'Silky Terrier', 'Golden Retriever', 'German Shepherd']
            predicted_name = f"Nom prédit par ViT-B16 : {breed_names[predicted_class_vit_b16]}"

            # Affichage formaté du résultat
            st.markdown(f"<h2 style='text-align: center; color: blue;'>{predicted_name}</h2>", unsafe_allow_html=True)
        else:
            st.error("Le modèle ViT-B16 n'est pas chargé correctement.")







