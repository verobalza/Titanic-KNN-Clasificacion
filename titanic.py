#librerias masa de numeros y dataframes
import pandas as pd
import numpy as np 

#librerias de modelado y separacion de datos
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#libreria de escalamiento y  minimos y maximos
from sklearn.preprocessing import MinMaxScaler

#librerias de graficos
import matplotlib.pyplot as plt 
import seaborn as sns 


#cargar data
data = pd.read_csv('titanic.csv')

#mostrar data
print( data.info())
print(data.isnull().sum())



#limpieza de data y conversion de variables categoricas a numericas
def data_preprocess(df):
    df.drop(columns=["PassengerId", "Name","Ticket", "Cabin", "Embarked"], inplace=True)

    #rellenar valores nulos
    df['Fare']= df['Fare'].fillna(df['Fare'].median())
    fill_mising_age(df)

    #convertir variable categorica en numerica
    df['Sex'] = df['Sex'].map({'male':1, 'female':0})


    #vemos si tienen familiares a bordo
    df['familySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = np.where(df['familySize']==0,1,0)

    #creamos bins de fare    
    df['fareBin'] = pd.qcut(df['Fare'],4, labels = False)
    #creamos rangos de edad
    df['agebin'] = pd.cut(df['Age'], bins = [0,12,20,30,40,50,60, np.inf], labels = False)

    return df

def fill_mising_age(df):
    age_fill_map = {}
    for pclass in df['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()
    df['Age'] = df.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) else row['Age'], axis=1)

#Actualizamos datos
datos = data_preprocess(data)

# separacion de caracteristicas y variable objetivo
x = datos.drop(columns=['Survived'])
y= datos['Survived']

# division de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)

#escalamiento de datos
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def sintonizacion(x_train, y_train):
    parametros ={
        'n_neighbors': range(1, 21),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, parametros, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

best_model = sintonizacion(x_train, y_train)

def evaluar_modelo(modelo, x_test, y_test):
    prediciones= modelo.predict(x_test)
    accurecy = accuracy_score(y_test, prediciones)
    matriz = confusion_matrix(y_test, prediciones)
    return accurecy, matriz
accuracy, matriz = evaluar_modelo(best_model, x_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print('Confusion Matrix:')
print(matriz)
