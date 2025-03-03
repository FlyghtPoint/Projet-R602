import pandas as pd
import random

# Lire le fichier CSV
df = pd.read_csv('ad_10000records.csv')

# Sélectionner les colonnes nécessaires
df = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Gender', 'Clicked on Ad']]

print("Distribution des valeurs 'Clicked on Ad':")
print(df['Clicked on Ad'].value_counts())

# Séparer les instances où "Clicked on Ad" (colonne 5) est 1 et 0
clicked = df[df['Clicked on Ad'] == 1].sample(n=1000, random_state=1)
not_clicked = df[df['Clicked on Ad'] == 0].sample(n=1000, random_state=1)

# Combiner les deux échantillons
df_mini = pd.concat([clicked, not_clicked])

# Mélanger les lignes
df_mini = df_mini.sample(frac=1, random_state=1).reset_index(drop=True)

# Enregistrer dans un nouveau fichier CSV
df_mini.to_csv('ad_mini.csv', index=False)