import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Charger les modèles
regression_model = joblib.load('random_forest_model_5f.pkl')
classification_model = joblib.load('best_classification_model_fixed.pkl')

df = pd.read_csv('kc_house_data.csv')  # Assurez-vous que ce fichier est présent

df["date"] = pd.to_datetime(df["date"])
df["year_sold"] = df["date"].dt.year
df["month_sold"] = df["date"].dt.month
df.drop("date", axis=1, inplace=True)  # On supprime la colonne originale

# Interface Streamlit
st.title('🏡 Prédiction & Classification des Maisons à Seattle')
st.write("Cette application permet de prédire le prix des maisons et de les classer en catégories.")

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio("Aller à", ['🏠 Analyse des données', '💰 Prédiction du Prix', '📊 Classification'])

if page == '🏠 Analyse des données':
    st.header('📊 Visualisation des données')
    st.write("Voici quelques statistiques sur le dataset des maisons :")
    st.dataframe(df.describe())
    
    # Distribution des prix
    st.subheader('Distribution des Prix')
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    # Corrélation
    st.subheader('Matrice de Corrélation')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("Quelques maisons en image :")
    st.image('images/house_image.jpg', caption='Exemple de maison')  # Ajoutez une image dans votre projet

elif page == '💰 Prédiction du Prix':
    st.header('💰 Prédiction du Prix des Maisons')
    
    # Saisie utilisateur
    bedrooms = st.number_input('Nombre de chambres', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Nombre de salles de bain', min_value=1, max_value=10, value=2)
    sqft_living = st.number_input('Surface habitable (sqft)', min_value=500, max_value=10000, value=1500)
    floors = st.number_input('Nombre d\'étages', min_value=1, max_value=3, value=1)
    zipcode = st.selectbox('Code postal', options=sorted(df['zipcode'].unique()))
    condition = st.slider('Condition de la maison (1=Très mauvais, 5=Très bon)', 1, 5, 3)
    grade = st.slider('Qualité de la maison (1=Bas, 13=Très élevé)', 1, 13, 7)
    yr_built = st.number_input('Année de construction', min_value=1900, max_value=2023, value=2000)
    yr_renovated = st.number_input('Année de rénovation', min_value=1900, max_value=2023, value=2020)
    
    if st.button('Prédire le Prix'):
        input_data = np.array([[sqft_living, bathrooms, bedrooms, floors, zipcode, condition, grade, yr_built, yr_renovated]])
        price = regression_model.predict(input_data)
        st.success(f'🏠 Le prix estimé de la maison est : ${price[0]:,.2f}')

elif page == '📊 Classification':
    st.header('📊 Classification des Maisons')
    
    # Transformation des prix en classes avec 4 intervalles de prix
    st.write("Les maisons sont classées en 4 catégories :")
    st.write("- 🟢 Bas Prix (< 300K$)\n- 🟡 Moyen Prix (300K$ - 800K$)\n- 🔴 Haut Prix (800K$ - 1.5M$)\n- 🔥 Très Haut Prix (> 1.5M$)")
    
    sqft_living = st.slider('Surface habitable (sqft)', 500, 10000, 1500)
    bathrooms = st.slider('Nombre de salles de bain', 1, 10, 2)
    bedrooms = st.slider('Nombre de chambres', 1, 10, 3)
    floors = st.slider('Nombre d\'étages', 1, 3, 1)
    zipcode = st.selectbox('Code postal', options=sorted(df['zipcode'].unique()))
    condition = st.slider('Condition de la maison (1=Très mauvais, 5=Très bon)', 1, 5, 3)
    grade = st.slider('Qualité de la maison (1=Bas, 13=Très élevé)', 1, 13, 7)
    yr_built = st.number_input('Année de construction', min_value=1900, max_value=2023, value=2000)
    yr_renovated = st.number_input('Année de rénovation', min_value=1900, max_value=2023, value=2020)
    
    # Mise à jour de la classification avec 4 intervalles de prix
    if st.button('Classer la maison'):
        input_data = np.array([[sqft_living, bathrooms, bedrooms, floors, zipcode, condition, grade, yr_built, yr_renovated]])
        category = classification_model.predict(input_data)[0]
        
        if category == 0:
            st.success('🏠 Cette maison est classée 🟢 Bas Prix')
        elif category == 1:
            st.warning('🏠 Cette maison est classée 🟡 Moyen Prix')
        elif category == 2:
            st.error('🏠 Cette maison est classée 🔴 Haut Prix')
        else:
            st.error('🏠 Cette maison est classée 🔥 Très Haut Prix')
