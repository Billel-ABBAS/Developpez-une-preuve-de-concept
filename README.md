# Projet 7 : Développez une preuve de concept

## Description
Ce projet de classification d’images a été réalisé dans le cadre d’un test technique pour un recrutement chez **DataSpace**, afin d'évaluer si un modèle de **Vision Transformer (ViT)**, basé sur des technologies récentes, peut surpasser les performances d'un modèle **CNN** plus classique, comme **Xception**, pour classifier des images de races de chiens.

Accédez à la démo de l’application Streamlit pour **la prédiction de la race de chiens avec le modèle ViT** déployé sur MLflow et Streamlit Cloud, et dont le modèle est stocké dans un bucket AWS :  
[**Lien vers l'application Streamlit**](https://openclassrooms-developpez-une-preuve-de-concept-hyyq6ey4sq43as.streamlit.app/)

## Objectifs
- **Explorer la performance comparative** entre un modèle CNN, Xception, et un Vision Transformer pour une tâche de classification d'images.
- **Appliquer des techniques de fine-tuning et de transfert d'apprentissage** sur un dataset de petite taille afin de maximiser la capacité de généralisation des modèles.
- **Tester l’efficacité des Vision Transformers** pour traiter des images de manière globale, en capturant le contexte de l'image, par rapport aux CNN qui se focalisent sur des caractéristiques locales.

## Données
Le projet utilise **827 images** issues du **Stanford Dogs Dataset**, représentant 5 races de chiens, avec une extension pour ajouter deux nouvelles races et augmenter la diversité :
- **Races sélectionnées** : Berger Allemand, Silky Terrier, Golden Retriever, Japanese Spaniel, et English Foxhound.

## Méthodologie
1. **Modèle Xception (CNN)** : Utilisé comme point de référence (baseline) et optimisé via du fine-tuning pour adapter ses connaissances aux cinq classes du dataset.
2. **Modèle ViT (Vision Transformer)** :
   - **Entraînement from scratch** : Testé pour évaluer la capacité d’apprentissage du ViT sur un petit dataset, mais les performances sont restées limitées, démontrant que ce modèle nécessite davantage de données.
   - **Transfert d'apprentissage** : Utilisation d’un modèle ViT pré-entraîné sur un large jeu de données (ImageNet-21K) et ajusté pour l’adaptation aux 5 classes du dataset, tirant ainsi parti de l’apprentissage initial tout en l’adaptant au problème spécifique.

## Résultats
- **Xception (CNN)** : Atteint **97% de précision** après fine-tuning, démontrant son efficacité pour les tâches de classification où les détails locaux sont essentiels.
- **ViT (Vision Transformer)** :
   - **From scratch** : Les résultats ont été limités sur ce petit dataset, illustrant que le ViT n’atteint son plein potentiel qu’avec des volumes de données plus importants.
   - **Transfert d’apprentissage** : Le modèle a atteint **99% de précision**, surpassant Xception, et prouvant sa capacité à capter les relations globales au sein de l'image.

## Conclusion
Le **Vision Transformer en transfert d'apprentissage** s’est révélé plus performant que le modèle Xception, atteignant une précision de 99% contre 97% pour Xception. Cela dit, le ViT est plus exigeant en termes de ressources de calcul. Xception reste un modèle robuste et efficace pour les tâches de classification plus conventionnelles, tandis que le ViT s’impose pour les analyses nécessitant une compréhension globale de l’image.

### Pistes d'amélioration
- **Augmenter le volume de données** pour affiner les performances, en particulier pour le ViT en entraînement from scratch.
- **Explorer des modèles hybrides**, tels que les **Convolutional Vision Transformers (CvT)**, afin de combiner les avantages des CNN et des transformers.
- **Tester le Swin Transformer en transfert d’apprentissage** pour voir s'il peut offrir un compromis entre attention locale et globale, tout en optimisant l’efficacité de calcul.

---

En résumé, ce projet met en évidence les forces et limites des deux approches : **Xception** pour une analyse fine des détails et **Vision Transformer** pour une compréhension contextuelle. Des étapes d’amélioration futures pourront enrichir les capacités de généralisation de ces modèles pour des tâches de classification d’images.
