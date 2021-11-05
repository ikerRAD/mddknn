import numpy as np
import pandas as pd
import statistics as st
import math
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


    


    

def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21
    mtcars.mpg[ix_consumo_alto] = 1
    mtcars.mpg[~ix_consumo_alto] = 0
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1)
    # AÃ±adimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    
    test, train = splitTrainTest(mtcars_normalizado, 0.75)

    # Separamos las labels del Test. Es como si no nos las dieran!!
    testT = test.loc[:, test.columns != 'mpg']
    testR = test.loc[:, test.columns == 'mpg']
    
    print(testT)
    print(testR)
    # Predecimos el conjunto de test
    
    true_labels = 0
    predicted_labels = 0
    for i in range(len(testT)):
        
        estimado = knn(testT.iloc[i,:], train, 3)
        predicted_labels +=1
        if estimado == testR[i]:
            true_labels +=1

    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    print(accuracy(true_labels, predicted_labels))

    # Algun grafico? Libreria matplotlib.pyplot
    return(0)

# FUNCIONES de preprocesado
def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x):
    return((x-st.mean(x))/st.variance(x))

# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    
    v = np.random.rand(len(data))
    mask = v>0.75 #Split de test
    test = data[mask]
    train = data[~mask]

    return(test, train)

def kFoldCV(data, K):
    """
    Takes a pandas dataframe and the number of folds of the CV
    YOU CAN USE THE sklearn KFold function here
    How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    return()

# FUNCION modelo prediccion
def knn(newx, data, K):
    
    
    trainT = data.loc[:, data.columns != 'mpg']
    trainR = data.loc[:, data.columns == 'mpg'] 

    return knnperform(K, newx, trainT, trainR)


#Función para ejecutar el knn
def knnperform(k, caso, dataset, clases):
    
    dists = []
    distssorted = []
    #calculamos las distancias
    for dato in dataset:
        distancia = euclideanDistance2points(caso,dato)
        dists.append(distancia)
        distssorted.append(distancia)
    distssorted.sort()
    
    #obtenemos los índices de los k vecinos más cercanos
    indexes = []
    for neigh in range(k):
        #obtenemos el índice de los k elementos más cercanos
        index = dists.index(distssorted[neigh])
        indexes.append(index)
        
        #marcamos la distancia del elemento usado para que no se repita
        dists[index] = -1
    
    #clases estimadas
    clasesm = [clases[i] for i in indexes]
    
    #obtenemos las diferentes clases posibles
    cldif = [clasesm[0]]
    for c in clasesm:
        if c not in cldif:
            cldif.append(c)
            
    #obtenemos la clase con mayor número de votos
    clasem = cldif[0]
    mayoritaria = clasesm.count(clasem)
    for c in cldif:
        nueva = clasesm.count(c)
        if nueva > mayoritaria:
            mayoritaria = nueva
            clasem = c
    
    #devolvemos el resultado
    return clasem
    

def euclideanDistance2points(x,y):
    
    suma = 0
    for i in range(len(x)):
        suma = np.power(x.loc[i]-y.loc[i], 2)
        
    return np.sqrt(suma) 


# FUNCION accuracy
def accuracy(true, pred):
    return true/pred


if __name__ == '__main__':
    np.random.seed(25)
    main()



















