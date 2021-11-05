import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from pylab import rcParams


    


    

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

    # Predecimos el conjunto de test
    
    true_labels = 0
    predicted_labels = 0
    print("\nCasos de test:\n")
    for i in range(len(testT)):
        
        estimado = knn(testT.iloc[i,:], train, 3)
        predicted_labels +=1
        print("prueba ",i,", Clase estimada --> ",estimado," Clase real --> ",testR.iloc[i,0],"\n")
        if estimado == testR.iloc[i,0]:
            true_labels +=1

    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    print(accuracy(true_labels, predicted_labels))

    print("\n5-fold cross-validation:")
    print(kFoldCV(mtcars_normalizado, 5))
    
    print("\nleave-one-out cross-validation:")
    print(kFoldCV(mtcars_normalizado, len(mtcars_normalizado)))

    plots(mtcars)
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
    
    #we create an array of len(data) values to delimitate the folds
    if len(data) == K:
        v = list(range(K))
    else:
        #creamos un vector de indices uniformemente distribuidos
        provisional = list(range(K))
        mult = int(len(data)/K)
        provisional = provisional*mult        
        v = provisional + list(range(len(data)%K))
        
    #randomizamos v pero no perdemos elementos
    np.random.shuffle(v)
    
    #hacemos K validaciones
    tasa = 0
    for i in range(K):
        tasatmp = 0
        totales = 0
        indexes = []
        notindexes = []
        print("Fold-",i+1,":\n")
        #obtenemos los índices de test y de train
        for j in range(len(v)):
            if v[j] == i:
                indexes.append(j)
            else:
                notindexes.append(j)
                
        #hacemos test
        test = data.iloc[indexes,:]
        train = data.iloc[notindexes,:]
        
        testT = test.loc[:, test.columns != 'mpg']
        testR = test.loc[:, test.columns == 'mpg']
        
        #estimamos las muestras de test
        for t in range(len(testT)):           
            estimado = knn(testT.iloc[t,:], train, 3)
            totales +=1
            print("prueba ",t,", Clase estimada --> ",estimado," Clase real --> ",testR.iloc[t,0],"\n")
            if estimado == testR.iloc[t,0]:
                tasatmp +=1
        
        #calculamos valores
        tasa += accuracy(tasatmp, totales)
        
    #devolvemos la tasa media
    return tasa/K

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
    for i in range(len(dataset)):
        dato = dataset.iloc[i,:]
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
    clasesm = []
    for i in indexes:
        clasesm.append(clases.iloc[i,0])
    
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
        
        suma = np.power(x[i]-y[i], 2)
        
    return np.sqrt(suma) 


# FUNCION accuracy
def accuracy(true, pred):
    return true/pred

#FUNCION graficos
def plots(data):
    #obtenemos el split de las clases
    clase0 = data[data['mpg']==0]
    clase1 = data[data['mpg']==1]
    
    rcParams['figure.figsize'] = 15, 20
    
    fig,axs = plt.subplots(10,1)
    fig.suptitle('Features of differen classes')
    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95, wspace = 0.2, hspace = 0.9)
    
    for param in range(1,len(data.columns)-1):
        ax = axs[param-1]
        clase0[data.columns[param]].plot(kind='density', ax=ax, subplots=True, 
                                    sharex=False, color="red", legend=True,
                                    label=data.columns[param] + ' for Outcome = 0')
        clase1[data.columns[param]].plot(kind='density', ax=ax, subplots=True, 
                                     sharex=False, color="green", legend=True,
                                     label=data.columns[param] + ' for Outcome = 1')
        ax.set_xlabel(data.columns[param+1] + ' values')
        ax.set_title(data.columns[param+1] + ' density')
        ax.grid('on')
    plt.show()
    fig.savefig('densities.png')

if __name__ == '__main__':
    np.random.seed(25)
    main()



















