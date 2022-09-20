#Margarita Sanchez

import random as ran # los valores aleatorios
import numpy as np # Calculos matematicos
import time as tm # Se encarga de las interrupciones para la muestra por consola
import matplotlib.pyplot as plt # Muestra de los datos en grafica
import pygame

#                            Funciones
# Funcion de Activacion
sigmoide = (lambda x: 1 / (1 + np.e ** (-x)), lambda x: x * (1 - x)) # Tupla con las 2 funciones, sigmoide y la derivada de la sigmoide


# Funcion de coste
  
cuadraticoMedio = (lambda Yr, Yp: np.mean((Yp - Yr) ** 2), lambda Yp, Yr: (Yp - Yr))       #el primer valor de la tupla indica el error cuadratico medio (np.mean calcula el medio) y el segundo valor es su derivada (todo para la funcion de coste)

# Yp es la prediccion e Yr la salida real que obtuvo

class Neurona():
  #--------------------------------
  
  # Constructor de la Neurona
  def __init__(self, cantEntradas, w = None, b = 0):
    
    # Si ya tengo determinado el bias y el peso los igualo y salgo
    if w != None :
      self.__w = w
      self.__b = b
      self.cantEntradas = len(w)
      return
    else:
      self.cantEntradas = cantEntradas
      self.Aleatorizar() # Genera pesos y bias aleatorios
    return
  
  #--------------------------------
  
  #Aleatoriza todos los datos de la neurona
  def Aleatorizar(self):
    self.__w =[] # pesos                  
    self.__b = ran.random()*2 -1   # bias 
    
    for i in range(self.cantEntradas):
      self.__w.append(ran.random()*2 -1) # escoge un valor aleatorio entre -1 y 1
  
  #--------------------------------
  
  # Muta a la Neurona
  def Mutar(self, probabilidad):
    
    aleatorio = ran.random()
    if(aleatorio < probabilidad) : # Primero actualizo el bias.
      self.__b += ran.random()*2 -1 # escoge un valor aleatorio entre -1 y 1 para el bias
      if(self.__b > 1): 
         self.__b = 1
      else:  
        if(self.__b < -1):
          self.__b = -1
    for i in range(self.cantEntradas): # Recorre cada peso
      aleatorio = ran.random()
      if(aleatorio < probabilidad) :
        self.__w[i] += ran.random()*2 -1 # escoge un valor aleatorio entre -1 y 1 para el resto de pesos
        if(self.__w[i] > 1): 
          self.__w[i] = 1
        else:  
          if(self.__w[i] < -1):
            self.__w[i] = -1
          
  
  #--------------------------------
  
  # Permite modificar los pesos y el bias
  def __ModificarWB(self, w, b):
    self.__w = w
    self.__b = b
  
  
  
  #--------------------------------
  
  
  # Es la funcion que corresponde a la salida en crudo de la neurona (la suma ponderada)
  def __SumaPonderada(self, x):
    total = 0
    
    for i in range(len(self.__w)):
      total += self.__w[i] * x[i]
    
    total += self.__b
    
    return total
  
  
  
  #--------------------------------        
  
  # Devuelve una copia de la neurona
  def Clon(self):
    return Neurona(None, self.__w.copy(), self.__b)
  
  
  #--------------------------------
  
  # Activacion de la neurona (devuelve la salida en crudo de la neurona)
  def Activar(self, x):
    salida = self.__SumaPonderada(x)
    return salida

class Capa():

  
  # Constructor de la clase Capa
  def __init__(self, cantNeuronas, cantEntradas, funcionDeActivacion, neuronas = None):
    
    self.funcionDeActivacion = funcionDeActivacion
    
    if neuronas != None : 
      
      self.__neuronas = neuronas
      return
    
    else:
      self.__neuronas = []
    
    
    # Agrega las neuronas a la lista
    for i in range(cantNeuronas):
      self.__neuronas.append(Neurona(cantEntradas))
  
    #print("cantidad de neuronas: ", len(self.__neuronas))
  
  #--------------------------------
  
  # Devuelve la cantidad de neuronas de esta capa
  def getCantNeuronas(self):
    return len(self.__neuronas)
  
  
  #--------------------------------
  
  # Devuelve la neurona indicada
  def getNeurona(self, pos):
    
    # Controla el rango
    if pos > len(self.__neuronas) or pos < 0:
      print("Fuera de rango en array '__neuronas' dentro de 'getNeurona' siendo pos = " + str(pos) + " y el largo: " + str(len(self.__neuronas)))
      return None
    
    
    return self.__neuronas[pos]
    
  #--------------------------------
  
  # Devuelve una copia de la capa
  def Clon(self):
    neuronasSalida = []
    funcionActivacionSalida = self.funcionDeActivacion
    
    for i in range(len(self.__neuronas)): 
      neuronasSalida.append(self.__neuronas[i].Clon())
      
    #Devuelve la nueva capa generada
    return Capa(None, None, funcionActivacionSalida, neuronasSalida)
    
    
    
  #--------------------------------
  # Activacion de la capa
  def Activar(self, x):
    
    salida = [] # la salida consta de un arreglo con las salidas de las neuronas
    
    for i in range(len(self.__neuronas)):
      # La salida de la capa pasando por la funcion de activacion elegida
      salida.append(self.funcionDeActivacion[0](self.__neuronas[i].Activar(x))) # a todas las neuronas se les da la misma entrada
      
    return salida  
  
  #--------------------------------   
  
  # Intenta mutar cada una de las neuronas de la capa
  def IntentaMutar(self, probabilidad):
    
    # Recorre cada neurona
    for i in range(len(self.__neuronas)) :
        self.__neuronas[i].Mutar(probabilidad) # Intenta mutar todas las neuronas de la capa
    return


  #--------------------------------

class Red():
  
  
  # Constructor de la Red neuronal
  def __init__(self, neuronasPorCapa, activacionPorCapa, capas = None):
    
    # Si existe una capa quiere decir que se instancia a partir de capas establecidas
    if capas != None :
      
      self.capa = capas
      return
    
    
    else:
      self.capa = []

    for i in range(len(neuronasPorCapa)):
      #Si es la primer capa entonces es la entrada
      if i-1 < 0:
        self.capa.append(Capa(neuronasPorCapa[i], neuronasPorCapa[i], activacionPorCapa[i]))

      #Sino, es una de las siguientes capas
      else:
        self.capa.append(Capa(neuronasPorCapa[i], neuronasPorCapa[i-1], activacionPorCapa[i]))
        
  
  #--------------------------------
  
  def getCantNeuronas(self):
    total = 0
    for i in range(len(self.capa)):
      total += self.capa[i].getCantNeuronas()
      
    return total      
  #--------------------------------        
  
  # Devuelve una copia de esta red
  def Clon(self):
    capasSalida = []
    
    for i in range(len(self.capa)):
      capasSalida.append(self.capa[i].Clon())
    
    return Red(None, None, capasSalida)
  
  
  #-------------------------------- 
  #Activa la red neuronal
  
  def Activar(self, x):
    
    for i in range(len(self.capa)):
       x = self.capa[i].Activar(x) # cada iteracion se actualiza el valor de x
      
    return x # Devuelve la salida de la red
  
  #--------------------------------   
  
  # Muestra la topologia de la red
  def Mostrar(self):
    for i in range(len(self.capa)):
      print("La capa " + str(i) + " tiene " + str(self.capa[i].getCantNeuronas()) + " Neuronas")
      
  #--------------------------------   
  
  # Intenta mutar
  def Mutar(self, probabilidad):
    
    # Recorre cada capa
    for i in range(len(self.capa)):
        self.capa[i].IntentaMutar(probabilidad)
    return


  #--------------------------------

promedios = [] # Para mostrar los datos luego
mejor = []

class SeleccionNatural():
  
  #--------------------------------   
  
  # Poblacion es la cantidad de simulaciones en una generacion, generaciones es la cantidad de generaciones hasta que el programa termine (-1 para infinitas)
  def __init__(self,neuronasPorCapa, activacionPorCapa, poblacion = 2000, generaciones = -1):
    self.redes = []
    self.__poblacion = poblacion
    self.__generaciones = generaciones
    self.mejorGlobal = None # El mejor de todas las generaciones
    self.mejorPuntuacionGlobal = 0
    for i in range(poblacion):
      self.redes.append(Red(neuronasPorCapa, activacionPorCapa))
      
    
    return
  
  
  #--------------------------------   
  
  # Comienza 
  def Start(self, funcionDeCoste, probabilidadDeMutar = 0.01, versus = False): # versus indica si se jugará algo tipo tateti en el que se requieren 2 redes
    mejorRed = None
    if(self.__generaciones <= 0):
      i = 0
      while(True):
        self.__Mejora(funcionDeCoste, probabilidadDeMutar,i, versus)
        snake = Game(mejorRed)
        copiar = []
        snake.Start(mejorRed, copiar)
        MostrarMejorCadaGeneracion(copiar)
        i += 1
    else:
      for i in range(self.__generaciones):
        mejorRed = self.__Mejora(funcionDeCoste, probabilidadDeMutar,i, versus)        
        snake = Game(mejorRed)
        copiar = []
        snake.Start(mejorRed, copiar)
        MostrarMejorCadaGeneracion(copiar)
    return self.MejorGlobal
  
  #--------------------------------  
  
  # Checkea si se trata de un "versus" o no, y devuelve la puntuacion de la red
  def DecideVersus(self,i, funcionDeCoste, estadoRandom, versus = False):
    puntuacionRed = 0
    if versus:
      puntuacionRed = funcionDeCoste(self.redes[i], self.redes[ran.randint(0,len(self.redes)-1)], estadoRandom)
    else: 
      puntuacionRed = funcionDeCoste(self.redes[i], estadoRandom)
    
    return puntuacionRed
  
  #--------------------------------  
  
  def ElijeDosPadres(self, posiblesPadres):
    padresP = posiblesPadres.copy() # Para no alterar los valores originales
    padres = [None, None]
    
    for i in range(len(padres)):
      valor = ran.randint(0, len(padresP)-1)
      padres[i] = padresP[valor] # elije un padre de los 3 posibles
      padresP.pop(valor) # lo elimina así no es reelegido
    
    return padres
  
  #--------------------------------
  # Decide las 2 mejores redes y cruza la siguiente generacion
  def __Mejora(self, funcionDeCoste, probabilidadDeMutar, generacionNum = 0, versus = False):
    estadoRandom = ran.getstate() # Con esto me aseguro que toda la poblacion de una generación va a recibir el mismo desafío, es decir, el mismo tablero.
    
    maximo = self.DecideVersus(0, funcionDeCoste,estadoRandom, versus)
    segundoMaximo = self.DecideVersus(1, funcionDeCoste,estadoRandom, versus)
    
    redMax = self.redes[0].Clon()
    segundoRedMax = self.redes[1]
    
    cantIteraciones = len(self.redes)
    promedioPuntuacion = 0
    # Busco los 2 mejores
    #print("--------------------------------------------------------------------------")
    for i in range(cantIteraciones): # CORREGIR LA  ITERACION INECESARIA
      #print("random: " + str(ran.randint(0, 1000)))
      puntuacionRed = self.DecideVersus(i, funcionDeCoste,estadoRandom, versus)
      
      promedioPuntuacion += puntuacionRed # Actualiza el promedio
      if(puntuacionRed > maximo):
        segundoMaximo = maximo
        segundoRedMax = redMax
        
        maximo = puntuacionRed
        redMax = self.redes[i].Clon()
        
      else:
        if(puntuacionRed > segundoMaximo):
          segundoMaximo = puntuacionRed
          segundoRedMax = self.redes[i].Clon()
    
    
    # Actualizo el mejor global
    if(self.mejorGlobal == None or maximo > mejorPuntuacionGlobal): # Si no existe el mejor aún o bien el maximo actual supera al maximo global
      self.MejorGlobal = redMax.Clon() # Este es el nuevo mejorGlobal
      self.mejorPuntuacionGlobal = maximo
    
    promedioPuntuacion = promedioPuntuacion / self.__poblacion
    
    promedios.append(promedioPuntuacion)
    mejor.append(maximo)
    
    print("Generación: " + str(generacionNum) + " promedio: " + str(promedioPuntuacion) + " mejor: " + str(maximo))
    

    
    
    
    #Luego de encontrar los mejores la siguiente generacion es fruto de ellos
    
    
    posiblesPadres = [self.MejorGlobal, redMax, segundoRedMax]
    self.redes = []
    
    self.redes.append(self.MejorGlobal.Clon())
    for i in range(cantIteraciones-1):
      padre = self.ElijeDosPadres(posiblesPadres)
      redHijo = self.CrossOver(padre[0], padre[1]) # Cruza ambos padres
      redHijo.Mutar(probabilidadDeMutar) # Muta al hijo
      self.redes.append(redHijo) # Lo añade a la lista
    
    return redMax # Devuelve al mejor
  
  #--------------------------------  
  

  # cruza a las 2 redes en una proporcion aleatoria copiando primero la red1 y luego la red2, en una nueva red.
  # PRECONDICION: AMBAS REDES TIENEN LA MISMA TOPOLOGIA (exactamente iguales en el orden y cantidad de capas y neuronas)
  def CrossOver(self, red1, red2):
    capasHijo = []
    
    contador = 0
    aleatorio = ran.randint(0, red1.getCantNeuronas())
    
    
    # recorre las capas
    for i in range(len(red1.capa)):
      neuronasAux = []
      
      # recorre las neuronas
      recorrido = red1.capa[i].getCantNeuronas()
      for j in range(recorrido):
        if(contador < aleatorio):
          neuronasAux.append(red1.capa[i].getNeurona(j).Clon())
        else:
          neuronasAux.append(red2.capa[i].getNeurona(j).Clon())
        contador += 1 

      capasHijo.append(Capa(None, None, red1.capa[i].funcionDeActivacion, neuronasAux))
      
    return Red(None, None, capasHijo) #Devuelve una red que es crossover de los padres
  
  
    #--------------------------------

class Snake():
#-----------------------------------------------------------------------------------   
#  La direccion de la vibora y su movimiento están dandos por el pad numerico derecho del teclado
#     8
#   4   6      siendo 4 izquierda, 6 derecha, 2 abajo y 8 arriba
#     2
#----------------------------------------------------------------------------------- 
  # Constructor de la calse snake
  def __init__(self, snake, red, pasos = 200):
    self.posiciones = snake # La vibora y las posiciones que ocupa su cuerpo
    self.cola = 1 # el largo de la cola de la vibora
    self.direccion = 6
    self.direccionProhibida = 4
    self.pasos = pasos # Si pasos llega a 0, la serpiente muere
    self.red = red
    self.muerta = False
    self.tiempoViva = 0
#----------------------------------------------------------------------------------- 
  # Devuelve la posicion de la cabeza
  def Cabeza(self):
    return self.posiciones[0]
  
#----------------------------------------------------------------------------------- 
  # Devuelve la posicion de la cola
  def Cola(self):
    return self.posiciones[len(self.posiciones)-1]
  
#----------------------------------------------------------------------------------- 
  
  # Devuelve la direccion opuesta a la direccion recibida
  def DireccionOpuesta(self, direccion):
    
    if(direccion == 8):
      return 2
    if(direccion == 2):
      return 8
    if(direccion == 6):
      return 4
    if(direccion == 4):
      return 6
  
    print("direccion incorrecta")
    return direccion
  
#----------------------------------------------------------------------------------- 

  def Mover(self, direccion):
    
    if(direccion == self.direccionProhibida): # Si se quiere mover hacia donde no puede, es como si se moviera hacia adelante
      direccion = self.DireccionOpuesta(direccion)
      
    #print("direccion: " + str(direccion) + " prohibida: " + str(self.direccionProhibida))
    #print("Estaba: " + str(self.posiciones[0]))  
    # Decide la direccion y actualiza su cabeza
    if(int(direccion) == 8):
      self.posiciones.insert(0, [self.posiciones[0][0]-1, self.posiciones[0][1]]) # self.posiciones[PARTE DEL CUERPO A MOVER][0 = Y, 1 = X]
      self.direccionProhibida = 2 # la direccion hacia la que luego no puede moverse
    else:
      if(int(direccion) == 6):       
        self.posiciones.insert(0, [self.posiciones[0][0], self.posiciones[0][1]+1])
        self.direccionProhibida = 4
      else:
        if(int(direccion) == 2):
          self.posiciones.insert(0, [self.posiciones[0][0]+1, self.posiciones[0][1]])
          self.direccionProhibida = 8
        else:
          if(int(direccion) == 4):
            self.posiciones.insert(0, [self.posiciones[0][0], self.posiciones[0][1]-1])
            self.direccionProhibida = 6
      
      
    #print("Se mueve: " + str(self.posiciones[0]))  
    self.direccion = direccion
    
    
    return self.posiciones[0]
    
#----------------------------------------------------------------------------------- 

  # Elimina la cola para generar el efecto de movimiento  
  def EliminarCola(self):
    self.posiciones.pop(len(self.posiciones)-1)
    return
#----------------------------------------------------------------------------------- 

  # Agrega tiempo de vida
  def AgregarTiempoViva(self, cant = 1):
    self.tiempoViva += cant
    return
#----------------------------------------------------------------------------------- 

  # Resta pasos (al moverse)
  def RestarPasos(self, cant = 1):
    self.pasos -= cant
    return
  # Agrega pasos (al comer)
  def AgregarPasos(self, cant = 50):
    self.pasos += cant
    return
#-----------------------------------------------------------------------------------

class Game():

#----------------------------------------------------------------------------------- 
# Dentro del tablero los numeros significan:  
# 0- es un espacio vacio
# 1- es un espacio ocupado por la vibora
# 2- es un espacio ocupado por un muro
# 3- es un espacio ocupado por comida
#----------------------------------------------------------------------------------- 

  # Constructor de la clase.
  def __init__(self, red, dimensiones = [11,11]):
    if(dimensiones[0] < 6 or dimensiones[1] < 6):
      print("la dimension no puede ser menor a 6")
      dimensiones = [6,6]
    self.tablero = np.zeros((dimensiones[0], dimensiones[1]))
    self.InicializaTablero(red)
    self.ColocarComida()
    
#-----------------------------------------------------------------------------------    

  # Inicializa el tablero colocando los elementos necesarios al comienzo
  def InicializaTablero(self, red):
    for i in range(len(self.tablero)):
      for j in range(len(self.tablero[0])):
        if(i == 0 or i == len(self.tablero)-1 or j == 0 or j == len(self.tablero[0])-1):
          self.tablero[i][j] = 2
    
    
    x = int(len(self.tablero)/2)
    y = int(len(self.tablero[0])/2)
    self.tablero[y][x] = 1
    self.tablero[y][x-1] = 1
    self.tablero[y][y-2] = 1
    
    self.snake = Snake([[y,x],[y,x-1],[y,x-2]], red) # la vibora
    
    return
   
#----------------------------------------------------------------------------------- 

  # Muestra el tablero por consola
  def MostrarTableroConsola(self):
    for i in range(len(self.tablero)):
      print()
      for j in range(len(self.tablero[0])):
        if(self.tablero[i][j] == 0):
          print(" ", end = "  ")
        else:  
          print(int(self.tablero[i][j]), end = "  ")
        
    return    
  
  
#----------------------------------------------------------------------------------- 

  # Coloca la comida en el tablero
  def ColocarComida(self):
    posicionesPermitidas = []
    for i in range(len(self.tablero)):
      for j in range(len(self.tablero[0])):
        if(self.tablero[i][j] == 0):
          posicionesPermitidas.append([i,j])
        
    #for i in range(len(posicionesPermitidas)):
      #print(posicionesPermitidas)
        
        
    if(len(posicionesPermitidas) > 0):
      aux = posicionesPermitidas[ran.randint(0,(len(posicionesPermitidas)-1))] # Posicion aleatoria entre las posiciones permitidas
      self.comida = aux
      self.tablero[aux[0]][aux[1]] = 3 # actualiza la nueva comida
    else:  
        self.snake.muerta = True
        print("Terminó la partida, muere.")
    return
    
#----------------------------------------------------------------------------------- 

  # Se encarga del movimiento en el tablero y de hacer mover a la serpiente
  def Mover(self):
    
    if(self.snake.pasos <= 0): # Si se quedó sin pasos
      self.snake.muerta = True
      return
    
    movimiento = self.Red() # le pide el movimiento a la red
    
    nuevaPosCabeza = self.snake.Mover(movimiento)
  
    self.snake.RestarPasos()
    
    if(self.tablero[nuevaPosCabeza[0]][nuevaPosCabeza[1]] == 0 or self.tablero[nuevaPosCabeza[0]][nuevaPosCabeza[1]] == 3): # Si puede moverse en esa direccion
      cola = self.snake.Cola()
      
      
      
      if(self.tablero[nuevaPosCabeza[0]][nuevaPosCabeza[1]] == 3): # Si comio
          self.ColocarComida()
          self.snake.AgregarPasos()
          self.snake.cola += 1
      else: # Sino, elimino la cola (si comio pareceria que crece)
        self.tablero[cola[0]][cola[1]] = 0
        self.snake.EliminarCola()
        
      self.tablero[nuevaPosCabeza[0]][nuevaPosCabeza[1]] = 1
      
      
      
    else: #sino, murió
      self.snake.muerta = True
      
    #print("pasos: " + str(self.snake.pasos) + " " + " cola:" + str(self.snake.cola))    
    return
    
#----------------------------------------------------------------------------------- 

  # Calcula la puntuación de esta partida y la devuelve
  
  
  def FuncionDeCoste(self):
      puntuacion = 0
      if(self.snake.cola < 10): 
         puntuacion = int(self.snake.tiempoViva/10) * np.power(2, self.snake.cola) # el tiempo viva/10 + 2^cola
            
      else: # Para no tener numeros tan grandes
        puntuacion = self.snake.tiempoViva * self.snake.tiempoViva;
        puntuacion *= np.power(2, 10);
        puntuacion *= (self.snake.cola - 9);
      
      return puntuacion 


#-----------------------------------------------------------------------------------   

  def ObtieneDatosDir(self,tablero, direccion = [0,0]): # La direccion se envia con -1, 0 o 1 según hacia que direccion se desea mover
    
    pos = self.snake.Cabeza().copy() # Se hace una copia para no alterar la posicion real
    
    # las mas cercanas a la hormiga
    
    salida = False
    encontroSuCola = False    
    
    
    
    inputRed = [0, 0, 0]
    distancia = 0
    while(not salida):
      pos[0] += direccion[0]
      pos[1] += direccion[1]
      
      distancia += 1 # colocar debajo en la formula 1 - distancia * valorEnDistancia
      
      if(tablero[pos[0]][pos[1]] == 2): # La distancia entre su cabeza y el muro en esta direccion
        salida = True # Si encontró el muro sale (ya que no verá mas allá)
        inputRed[0] =  1/ distancia 
      else:   
        if(tablero[pos[0]][pos[1]] == 3): # Si ve o no comida en esa direccion
          inputRed[1] = 1 
            
        else:
          if((not encontroSuCola) and tablero[pos[0]][pos[1]] == 1): # La distancia entre su cabeza y el cuerpo en esta direccion
            encontroSuCola = True
            inputRed[2] = 1/ distancia 
              

      
    return inputRed 
    
#----------------------------------------------------------------------------------- 

  # pide los datos de la direccion y los añade a la salida
  def __Unifica(self, tablero, salida, direccion = [0,0]):
    datosDir = self.ObtieneDatosDir(tablero, direccion)
    
    for i in range(len(datosDir)):
      salida.append(datosDir[i])
    
#-----------------------------------------------------------------------------------  
  # Obtiene los valores de las direcciones para la red neuronal
  def ObtenerDatosRed(self):
    salida = []
    
     # 2 8 6 9 3 4 7 1 
    self.__Unifica(self.tablero, salida, [0,1]) # 6
    self.__Unifica(self.tablero, salida, [0,-1]) # 4
    self.__Unifica(self.tablero, salida, [1,0]) # 2
    self.__Unifica(self.tablero, salida, [1,1]) #  3
    self.__Unifica(self.tablero, salida, [1,-1]) #  1
    self.__Unifica(self.tablero, salida, [-1,0]) # 8 
    self.__Unifica(self.tablero, salida, [-1,1]) # 9 
    self.__Unifica(self.tablero, salida, [-1,-1]) # 7 
    
   
    
    return salida
#----------------------------------------------------------------------------------- 

  # Comunicacion entre la red y el juego, devuelve el movimiento elegido por la red
  def Red(self):
    if(self.snake.red != None):
      inputRed = self.ObtenerDatosRed()
      salida = self.snake.red.Activar(inputRed)
      mayor = 0
      for i in range(4): # Elije al mayor de los 4
        if(salida[mayor] < salida[i]):
          mayor = i
          
      mayor = (mayor+1) * 2; # se multiplica por 2 para transformar el uno en 2, el dos en 4, el tres en 6 y el cuatro en 8 y convertirlos así en el codigo del movimiento
      return mayor; # devuelve 2, 4, 6 u 8
    else:
      self.MostrarTableroConsola()
      return int(input())
#----------------------------------------------------------------------------------- 
  # Esta función es la encargada de la ejecución del juego
  def Start(self, red, copiarMov, mostrarCopiar = True): # Copiar Mov será una lista pasada por referencia que contendra las posiciones del tablero en cada iteración para luego mostrar los datos en caso de ser la mejor
    while(not self.snake.muerta): # Mientras la serpiente no esté muerta
      #self.MostrarTableroConsola()
      #print(self.ObtenerDatosRed())
      if(mostrarCopiar):
        copiarMov.append(self.tablero.copy())
      self.snake.AgregarTiempoViva()
      self.Mover()

    
    return self.FuncionDeCoste()


#----------------------------------------------------------------------------------- 
  def ImprimirDatos(self):
    print("Cola: " + str(self.snake.cola) + " tiempo viva: "+ str(self.snake.tiempoViva)  + " pasos: " + str(self.snake.pasos) + " Puntuacion: " + str(resultado))
#----------------------------------------------------------------------------------- 

#----------------------------------------------------------------------------------- 

#----------------------------------------------------------------------------------- 

#-----------------------------------------------------------------------------------

def FuncionCoste(red, estadoRandom):
  
  ran.setstate(estadoRandom)
  snake = Game(red)
  copiar = []
  puntuacion = snake.Start(red, copiar, False)
  ran.setstate(estadoRandom)
  return puntuacion


def MostrarUltimaRed(copiar):
  for k in range(len(copiar)):
    tablero = copiar[k]
    print()
    print()
    for i in range(len(tablero)):
        print()
        for j in range(len(tablero[0])):
          if(tablero[i][j] == 0):
            print(" ", end = "  ")
          else:  
            print(int(tablero[i][j]), end = "  ")


            
            
#pygame.init()

dimensionesVentana = 660 # El tamaño de la ventana debe ser potencia de 10
win = pygame.display.set_mode((dimensionesVentana,dimensionesVentana))
pygame.display.set_caption("Snake")

# La division es una division entera
dimensiones = dimensionesVentana//11 # porque cada posicion es un cuadrado solamente requiere un unico valor de dimension. El calculo es para acomodar 


# Muestra a los mejores de cada generación jugar una partida (al ser una partida nueva será distinta a la que jugaron)
def MostrarMejorCadaGeneracion(copiar):
  for k in range(len(copiar)):
    win.fill((0,0,0))
    tablero = copiar[k]
    
    pygame.time.delay(100) # tiempo entre frames en milisegundos (100ms = 0.1s)

    
    for i in range(len(tablero)):
      for j in range(len(tablero[0])):
        if(tablero[i][j] == 1):
          pygame.draw.rect(win, (0,255,0), (i*dimensiones, j*dimensiones, dimensiones, dimensiones))  #This takes: window/surface, color, rect 
        if(tablero[i][j] == 2):
          pygame.draw.rect(win, (130,130,130), (i*dimensiones, j*dimensiones, dimensiones, dimensiones))  #This takes: window/surface, color, rect 
        if(tablero[i][j] == 3):
          pygame.draw.rect(win, (255,0,0), (i*dimensiones, j*dimensiones, dimensiones, dimensiones))  #This takes: window/surface, color, rect 
        
    pygame.display.update() # This updates the screen so we can see our rectangle
    
  pygame.time.delay(1000) # tiempo entre generaciones

neuronasPorCapa = [24, 16, 4] # topologia
activacionPorCapa = [sigmoide, sigmoide, sigmoide] # las activaciones capa a capa

poblacionPorGeneracion = 2000 # Cambiar este valor para alternar la población utilizada en una generación
cantGeneraciones = 100 # Cambiar este valor para alternar la cantidad de generaciónes maximas


seleccionNatural = SeleccionNatural(neuronasPorCapa,activacionPorCapa, poblacionPorGeneracion, cantGeneraciones)

mejorRed = seleccionNatural.Start(FuncionCoste, 0.05, False) # el número por defecto( 0.01 )  es la probabilidad de mutar al cruzar dos especies (colocar numero entre 0 y 1 inclusive.)  

#snake = Game(mejorRed)
#copiar = []
#snake.Start(mejorRed, copiar)


  
pygame.quit() 
#plt.plot(promedios) 
plt.plot(mejor)
plt.show()
plt.plot(promedios)
plt.show()

#Muestra la ultima partida
#MostrarUltimaRed(copiar)
