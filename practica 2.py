# Databricks notebook source
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS account;
# MAGIC
# MAGIC create table account
# MAGIC (account_id float,
# MAGIC district_id float,
# MAGIC frequency string,
# MAGIC `date` float)
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/data/account.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS client;
# MAGIC create table client
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/data/client.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS disp;
# MAGIC create table disp
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/data/disp.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS district;
# MAGIC create table district
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/data/district.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS loan;
# MAGIC create table loan
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/data/loan.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE view tabla_minable 
# MAGIC as
# MAGIC SELECT *
# MAGIC FROM 
# MAGIC (
# MAGIC 	SELECT client_id, district_id
# MAGIC 	     , if (SUBSTRING(birth_number, 3, 2) <= '50', 'hombre', 'mujer') as genero -- extrayendo el genero
# MAGIC     FROM 
# MAGIC 	    client
# MAGIC ) as c
# MAGIC    LEFT JOIN (
# MAGIC 				SELECT A1 as districtid
# MAGIC 					 , A2 as distrito
# MAGIC 					 , A3 as region
# MAGIC 					 , A4 as habitantes
# MAGIC 					 , A9 as num_ciudades
# MAGIC 					 , A10 as ratio_residentes_urbanos
# MAGIC 					 , A11 as Salario_promedio
# MAGIC 				FROM 
# MAGIC 					district
# MAGIC              ) as dis on dis.districtid = c.district_id
# MAGIC   INNER JOIN (
# MAGIC 				SELECT client_id as c_id , account_id, type as tipo_disposicion_cliente
# MAGIC                 FROM
# MAGIC 					disp
# MAGIC 			 ) as di on di.c_id = c.client_id
# MAGIC   INNER JOIN (
# MAGIC 				SELECT account_id as ac_id
# MAGIC 					 -- , district_id as district_id_cuenta -- info  a consideracion debido a que nos interesa el distrito del cliente
# MAGIC 					 -- , STR_TO_DATE(date, '%y%m%d') as fecha_creacion_cuenta -- conversion de la fecha
# MAGIC 					 , case
# MAGIC 						  when frequency = 'POPLATEK MESICNE' then 'emision mensual'
# MAGIC 						  when frequency = 'POPLATEK TYDNE' then 'emision semanal'
# MAGIC 						  when frequency = 'POPLATEK PO OBRATU' then 'emision despues de una transaccion'
# MAGIC 					    else 'no especifica'
# MAGIC 					    end as frecuencia_emision			
# MAGIC 				FROM 
# MAGIC 					account
# MAGIC 			 ) as ac on ac.ac_id = di.account_id
# MAGIC
# MAGIC  LEFT JOIN (
# MAGIC 				SELECT loan_id, account_id as accl_id
# MAGIC 					 , amount as monto_prestamo, duration as duracion_prestamo
# MAGIC 					 , payments AS pagos_mensuales
# MAGIC                      , status  as estado
# MAGIC 				FROM 
# MAGIC 					loan
# MAGIC              ) as ln on ln.accl_id = ac.ac_id
# MAGIC
# MAGIC ;

# COMMAND ----------

# MAGIC  %sql
# MAGIC select * from tabla_minable

# COMMAND ----------

# MAGIC %sql
# MAGIC select district_id, region, habitantes, ratio_residentes_urbanos, Salario_promedio, frecuencia_emision, 
# MAGIC monto_prestamo, duracion_prestamo, pagos_mensuales, estado from tabla_minable
# MAGIC

# COMMAND ----------

df = spark.sql(""" select district_id, region, habitantes, ratio_residentes_urbanos, Salario_promedio, frecuencia_emision,
monto_prestamo, duracion_prestamo, pagos_mensuales, estado from tabla_minable""")

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col
df = df.withColumn("district_id", col("district_id").cast("string"))
df.printSchema()

# COMMAND ----------

df_cl = df.na.drop()
df_cl.show()

# COMMAND ----------

df_cl.count()

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df_cl.groupBy(F.col('estado')).count().show()

# COMMAND ----------

from pyspark.sql.functions import when, col
df_cl = df_cl.withColumn(
    'candidato',
    when((df_cl['estado'] == 'A') | (df_cl['estado'] == 'C'), 'Candidato')
    .otherwise('No candidato')
)
df_cl.show()

# COMMAND ----------

df_final = df_cl.drop('estado')
df_final.show()

# COMMAND ----------

df_final.groupBy(F.col('candidato')).count().show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
# one hot encoding district_id
district_indexer = StringIndexer(inputCol= 'district_id', outputCol= 'districtIndex')
district_encoder = OneHotEncoder(inputCol = 'districtIndex', outputCol= 'districtVec')
# one hot encoding region
region_indexer = StringIndexer(inputCol='region', outputCol= 'regionIndex')
region_encoder = OneHotEncoder(inputCol = 'regionIndex', outputCol= 'regionVec')
# one hot encoding frecuencia_emision
frecuencia_indexer = StringIndexer(inputCol='frecuencia_emision', outputCol= 'frecuenciaIndex')
frecuencia_encoder = OneHotEncoder(inputCol = 'frecuenciaIndex', outputCol= 'frecuenciaVec')

# COMMAND ----------

df_final.columns

# COMMAND ----------

candidato_indexer = StringIndexer(inputCol= 'candidato', outputCol= 'candidatoIndex')
assembler = VectorAssembler(inputCols = ['districtVec', 'regionVec', 'frecuenciaVec',
                                        'habitantes',
                                        'ratio_residentes_urbanos',
                                        'Salario_promedio',
                                        'monto_prestamo',
                                        'duracion_prestamo',
                                        'pagos_mensuales'
                                        ], outputCol= 'features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
log_reg_candidato = LogisticRegression(featuresCol= 'features', labelCol='candidatoIndex')

# COMMAND ----------

pipeline = Pipeline(stages= [
    district_indexer,
    region_indexer,
    frecuencia_indexer,
    district_encoder,
    region_encoder,
    frecuencia_encoder,
    candidato_indexer,
    assembler, 
    log_reg_candidato])

# COMMAND ----------

train_data, test_data = df_final.randomSplit([0.7,0.3])
fit_model = pipeline.fit(train_data)
type(fit_model)

# COMMAND ----------

results = fit_model.transform(test_data)
type(results)

# COMMAND ----------

results.show(5)

# COMMAND ----------

results.toPandas()
results

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results.select('candidatoIndex', 'prediction').show(10)

# COMMAND ----------

auc = me_eval.evaluate(results)
print(auc)

# COMMAND ----------

training_summary = fit_model.stages[-1].summary
roc = training_summary.roc.toPandas()
import matplotlib.pyplot as plt
plt.plot(roc['FPR'], roc['TPR'])
plt.ylabel('True positive rate')
plt.xlabel('False postive rate')
plt.title('curva ROC')
plt.show()
