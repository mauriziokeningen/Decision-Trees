import pandas as pd
from matplotlib import pyplot as plt
import sklearn.tree as skl
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

# Adquiere los datos desde un archivo csv usando la biblioteca Pandas
dataframe = pd.read_csv(r'medicinas.csv')

# Hacer una copia del DataFrame original
dataframe_transformed = dataframe.copy()

# Preprocesamiento (Transformar valores categóricos a numéricos)
Edad = LabelEncoder()
Sexo = LabelEncoder()
PresionSanguinea = LabelEncoder()
Colesterol = LabelEncoder()
Medicamento = LabelEncoder()

dataframe_transformed['Edad'] = Edad.fit_transform(dataframe_transformed['Edad'])
dataframe_transformed['Sexo'] = Sexo.fit_transform(dataframe_transformed['Sexo'])
dataframe_transformed['PresionSanguinea'] = PresionSanguinea.fit_transform(dataframe_transformed['PresionSanguinea'])
dataframe_transformed['Colesterol'] = Colesterol.fit_transform(dataframe_transformed['Colesterol'])
dataframe_transformed['Medicamento'] = Medicamento.fit_transform(dataframe_transformed['Medicamento'])

# Prepara los datos
features_cols = ['Edad', 'Sexo', 'PresionSanguinea', 'Colesterol']
X = dataframe_transformed[features_cols]
y = dataframe_transformed.Medicamento

# Entrenamiento
tree = skl.DecisionTreeClassifier(criterion='gini')
tree.fit(X, y)

# Visualización
fig = plt.figure(figsize=(10, 10))
_ = plot_tree(tree, feature_names=features_cols, class_names=['A', 'B'], filled=True)

# Guardar el gráfico del árbol de decisión en un archivo
fig.savefig('arbol_decision.png')

# Probar el Modelo
dfprueba = pd.DataFrame()
dfprueba['Edad'] = [1]  # Mediana-Edad
dfprueba['Sexo'] = [0]  # F
dfprueba['PresionSanguinea'] = [2]  # Baja
dfprueba['Colesterol'] = [0]  # Normal

prediccion = tree.predict(dfprueba)

print('Resultado de la prueba')
print('**********************')
print('Con los datos')
print(dfprueba)
print('\nSe recomienda:')
if prediccion[0] == 0:
    print('A')
else:
    print('B')
print("**********************")
