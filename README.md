# 🧠 LLM Playground : Comprendre les Grands Modèles de Langage

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
</p>

## 📌 Présentation du Projet
Ce projet est une immersion technique dans le fonctionnement des **Large Language Models (LLM)**. J'ai conçu, entraîné et déployé un modèle de type **GPT-2** de **39,8 millions de paramètres**. 

L'objectif est de démontrer comment une architecture Transformer apprend des structures linguistiques, depuis la prédiction de tokens jusqu'à l'adoption de registres spécifiques via le **fine-tuning**.

🔗 **Accéder à l'application interactive :** HYPERLIEN

---

## 🏗️ Architecture & Spécifications techniques
Le modèle, baptisé **TinyGPT V4**, repose sur les caractéristiques suivantes :
* **Structure :** 10 couches (blocks) avec 8 têtes d'attention chacune.
* **Dimensions :** $d_{model} = 512$ et une expansion Feed-Forward à 2048.

---

## 🧪 Expérimentations & Fine-Tuning
L'application permet de comparer trois versions du modèle pour observer l'impact du dataset sur le style de génération :

| Version | Dataset d'entraînement | Style de sortie |
| :--- | :--- | :--- |
| **Base Model** | TinyStories (500k histoires) | Narratif enfantin  |
| **Wikipedia** | Simple English Wikipedia | Encyclopédique / Factuel |
| **TextBook** | Cosmopedia (80%) + TinyStories (20%) | Pédagogique / Éducatif |

>**Note sur le Catastrophic Forgetting :** Le passage à la version Wikipedia montre une perte de la capacité narrative initiale, un phénomène que j'ai atténué dans la version TextBook en mélangeant les datasets.

---
