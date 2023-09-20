# Databricks notebook source
# Parte 1 - Importamos las librerias necesarias.

# Importación de las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS facturas;
# MAGIC create table facturas
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/practicafinalmodulo4/factura.csv",  header "true", delimiter ",", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from facturas

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE view st 
# MAGIC as 
# MAGIC select Fecha, Totalconsigno as total
# MAGIC from facturas 

# COMMAND ----------

df = spark.sql("select Fecha, Totalconsigno as total from facturas")
df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.count()

# COMMAND ----------

df_clean = df.na.drop()
df_clean.show()

# COMMAND ----------

df_clean.count()

# COMMAND ----------

df_final = df_clean.toPandas()
df_final

# COMMAND ----------

df_final.info()

# COMMAND ----------

df_final['Fecha']=pd.to_datetime(df_final['Fecha'])
df_final.info()

# COMMAND ----------

# colocando la fecha como indice
df_final= df_final.set_index('Fecha')
df_final

# COMMAND ----------

# En caso de ser necesario ordenamos los registros por fecha.
df_final.sort_values('Fecha', inplace=True, ascending=True)
df_final

# COMMAND ----------

#suma por dia (colapso)
df_final=df_final.resample('D').sum()
df_final

# COMMAND ----------

df_final.describe()

# COMMAND ----------

# Visualizamos la informacion del conjunto de datos!
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
df_final['total'].plot()
plt.tight_layout()
plt.grid()
plt.show()

# Conclusiones:
# Zt = T + e + C + a
# Tendencia deterministica de tipo lineal creciente!
# Estacionalidad muy marcada!

# COMMAND ----------

# Descomposición de la serie de tiempo

import statsmodels.api as sm
import matplotlib.pyplot as plt
res = sm.tsa.seasonal_decompose(df_final,period=7)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

# COMMAND ----------

# Machine Learning , X  -->  y
# Train - Test

# En series de tiempo generalmente se hace un particion distinta!
# Se debe tener cuidado ya que el train spli es diferente!
train, test = df_final.iloc[0:-22], df_final.iloc[-22:len(df_final)]
print(len(train), len(test))

# COMMAND ----------

# Importante!
# En series de tiempo es necesario escalar las variable o caracteristicas!
# Para escalar puedes usar normalización,estandarización u otro tipo de escalado.
train_max = train.max()
train_min = train.min()

# COMMAND ----------

# Normalizamos los set de datos, train y test!
# Pueden utilizar la metodologia de escalamiento que deseen!
train_set_scaled = (train - train_min)/(train_max - train_min)
test_set_scaled = (test - train_min)/(train_max - train_min)

# COMMAND ----------

# Revisamos el escalado!
train_set_scaled.head(10)

# COMMAND ----------

train_set_scaled.shape

# COMMAND ----------

# Definimos la funcion de la arquitectura X - y.
# Funcion de los windows
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# COMMAND ----------

range(len(train_set_scaled) - 1)

# COMMAND ----------

(train_set_scaled.iloc[0:1].values).shape

# COMMAND ----------

v = train_set_scaled.iloc[0:1].values

# COMMAND ----------

Xs = []
Xs.append(v)

# COMMAND ----------

np.array(Xs).shape

# COMMAND ----------

np.array(Xs)

# COMMAND ----------

# Definimos un time_step y procedemos a crear los dataframes!
time_steps = 1 # Comencemos un vector unitario

X_train, y_train = create_dataset(train_set_scaled, train_set_scaled.total , time_steps)
X_test,  y_test  = create_dataset(test_set_scaled,  test_set_scaled.total, time_steps)

# COMMAND ----------

# reshape input debe ser 3D para las LSTM's: [samples, timesteps, features]
X_train.shape

# COMMAND ----------

X_train[47],X_train[48]

# COMMAND ----------

y_train[47]

# COMMAND ----------

# serie neuronal recurrente
X_train.shape[2]

# COMMAND ----------

# Importando de keras las librerias mas importantes!

from keras.models import Sequential # Arquitectura de red neuronal!
from keras.layers import Dense      # Capa densa!
from keras.layers import LSTM       # Capa recurrente
from keras.layers import Dropout    # Evita el overfitting (Inactiva algunas neuronas)

def lstm_architecture(X_data,rate_dropout):
    # Inicializando the RNN
    model = Sequential()

    # 1ra capa LSTM y Dropout para regularización.
    # input_shape (amplitude,1)
    # return_sequences = True, devolvera una salida por cada neurona de manera densa en false se unen las devoluciones para cada neurona
    model.add(LSTM(units = 250, return_sequences = True, input_shape=(X_data.shape[1], X_data.shape[2])))
    # 20% de las neuronas serán ignoradas durante el training (20%xNodos = 10)
    # Para hacer menos probable el overfiting
    model.add(Dropout(rate=rate_dropout))

    # 2da capa LSTM y Dropout para regularización.
    model.add(LSTM(units = 250, return_sequences = True))
    model.add(Dropout(rate=rate_dropout))

    # 3ra capa LSTM y Dropout para regularización.
    model.add(LSTM(units = 250, return_sequences = True))
    model.add(Dropout(rate=rate_dropout))

    # 4ta capa LSTM y Dropout para regularización.
    model.add(LSTM(units = 250, return_sequences = False))
    model.add(Dropout(rate=rate_dropout))

    # Capa de Salida!
    model.add(Dense(units = 1))

    # Resumen del modelo!
    model.summary()

    return model

# COMMAND ----------

import datetime
print('Iniciando a las: ', datetime.datetime.now())
print("...")

# Compiling the RNN
model_1 = lstm_architecture(X_data=X_train,rate_dropout=0.2)
model_1.compile(optimizer = 'adam', loss = 'mean_squared_error')

# COMMAND ----------

# Ejecutamos la RNN!

history = model_1.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    shuffle=False)

print("...")
print('Terminando a las: ', datetime.datetime.now())

# COMMAND ----------

# Revisamos algunos parametros de ajuste del modelo!
plt.plot(history.history['loss'], label='train')
plt.legend();
plt.show()

# COMMAND ----------

# Predecimos sobre la data de test!
y_pred = model_1.predict(X_test)

# COMMAND ----------

# Regresamos la informacion a los valores originales!
y_test = y_test*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores reales de test!
y_pred = y_pred*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores pronosticados para validar!
y_train = y_train*(train_max[0] - train_min[0]) + train_min[0]  # 652 valores de entrenamiento!

# COMMAND ----------

# Visualizamos los resultados!
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred.flatten(), 'r', marker='.', label="prediction")
plt.plot(np.arange(0, len(y_train)), y_train.flatten(), 'g', marker='.', label="history")
plt.ylabel('Count')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# COMMAND ----------

# Vemos algunos indicadores del ajuste!
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: ',rmse)

# Definimos y calculamos el MAPE (mean_absolute_percentage_error)
y_test, y_pred = np.array(y_test), np.array(y_pred)
print(f'MAPE: ',np.mean(np.abs((y_test - y_pred) / y_test)) * 100)


# COMMAND ----------

# MAGIC %md
# MAGIC Con time_step de 7

# COMMAND ----------

# Definimos un time_step y procedemos a crear los dataframes!
time_steps = 7 # Porque tengo data diaria y necesito estacionalidad de la semana!

X_train, y_train = create_dataset(train_set_scaled, train_set_scaled.total , time_steps)
X_test,  y_test  = create_dataset(test_set_scaled,  test_set_scaled.total, time_steps)

# COMMAND ----------

model_2 = lstm_architecture(X_data=X_train,rate_dropout=0.2)
model_2.compile(optimizer = 'adam', loss = 'mean_squared_error')

# COMMAND ----------

# Ejecutamos la RNN!

history = model_2.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    shuffle=False)

print("...")
print('Terminando a las: ', datetime.datetime.now())

# COMMAND ----------

# Revisamos algunos parametros de ajuste del modelo!
plt.plot(history.history['loss'], label='train')
plt.legend();
plt.show()

# COMMAND ----------

# Predecimos sobre la data de test!
y_pred = model_2.predict(X_test)

# Regresamos la informacion a los valores originales!
y_test = y_test*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores reales de test!
y_pred = y_pred*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores pronosticados para validar!
y_train = y_train*(train_max[0] - train_min[0]) + train_min[0]  # 652 valores de entrenamiento!

# Visualizamos los resultados!
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred.flatten(), 'r', marker='.', label="prediction")
plt.plot(np.arange(0, len(y_train)), y_train.flatten(), 'g', marker='.', label="history")
plt.ylabel('Count')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# COMMAND ----------

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: ',rmse)

# Definimos y calculamos el MAPE (mean_absolute_percentage_error)
y_test, y_pred = np.array(y_test), np.array(y_pred)
print(f'MAPE: ',np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC Con time_step de 14

# COMMAND ----------

# Definimos un time_step y procedemos a crear los dataframes!
time_steps = 14 # Porque tengo data diaria y necesito estacionalidad de cada 2 semanas!

X_train, y_train = create_dataset(train_set_scaled, train_set_scaled.total , time_steps)
X_test,  y_test  = create_dataset(test_set_scaled,  test_set_scaled.total, time_steps)

# COMMAND ----------

model_3 = lstm_architecture(X_data=X_train,rate_dropout=0.2)
model_3.compile(optimizer = 'adam', loss = 'mean_squared_error')

# COMMAND ----------

# Ejecutamos la RNN!

history = model_3.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    shuffle=False)

print("...")
print('Terminando a las: ', datetime.datetime.now())

# COMMAND ----------

# Revisamos algunos parametros de ajuste del modelo!
plt.plot(history.history['loss'], label='train')
plt.legend();
plt.show()

# COMMAND ----------

# Predecimos sobre la data de test!
y_pred = model_3.predict(X_test)

# Regresamos la informacion a los valores originales!
y_test = y_test*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores reales de test!
y_pred = y_pred*(train_max[0] - train_min[0]) + train_min[0]    # 100 valores pronosticados para validar!
y_train = y_train*(train_max[0] - train_min[0]) + train_min[0]  # 652 valores de entrenamiento!

# Visualizamos los resultados!
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred.flatten(), 'r', marker='.', label="prediction")
plt.plot(np.arange(0, len(y_train)), y_train.flatten(), 'g', marker='.', label="history")
plt.ylabel('Count')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# COMMAND ----------

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: ',rmse)

# Definimos y calculamos el MAPE (mean_absolute_percentage_error)
y_test, y_pred = np.array(y_test), np.array(y_pred)
print(f'MAPE: ',np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

# COMMAND ----------


