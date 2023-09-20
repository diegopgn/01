# Databricks notebook source
# MAGIC %md
# MAGIC *PRACTICA FINAL

# COMMAND ----------

# MAGIC %md
# MAGIC CREACION Y TRANSFORMACION DE TABLAS

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS producto;
# MAGIC create table producto
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/practicafinalmodulo4/Producto.csv",  header "true", delimiter ",", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from producto

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ventas;
# MAGIC create table ventas
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/practicafinalmodulo4/Ventas.csv",  header "true", delimiter ",", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ventas

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS mov;
# MAGIC create table mov
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/practicafinalmodulo4/Mov.csv",  header "true", delimiter ",", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from mov

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE view tab_min 
# MAGIC as
# MAGIC SELECT * 
# MAGIC FROM (select cliente_id, product_id, product_uom_qty, price_total
# MAGIC from ventas ) as v
# MAGIC LEFT JOIN (select `ID`, Preciodeventa, Cantidadamano, Nombre from producto) as p on v.product_id = p.`ID`
# MAGIC LEFT JOIN (SELECT product_id as pid , product_qty, location_dest_id from mov) as m on m.pid = v.product_id 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tab_min

# COMMAND ----------

# MAGIC %sql
# MAGIC select Nombre, price_total, product_uom_qty, cliente_id, Cantidadamano, product_qty, location_dest_id from tab_min

# COMMAND ----------

# MAGIC %sql
# MAGIC describe tab_min

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC -- resumen de las transacciones por cuenta
# MAGIC DROP TABLE IF EXISTS tm;
# MAGIC CREATE TABLE tm
# MAGIC AS
# MAGIC SELECT 
# MAGIC     cliente_id,
# MAGIC     MAX(Nombre) AS Nombre,
# MAGIC     AVG(price_total) AS precio_total,
# MAGIC     AVG(product_uom_qty) AS prod_vendido,
# MAGIC     AVG(Cantidadamano) AS stock,
# MAGIC     SUM(product_qty) as prod_mov,
# MAGIC     MAX(location_dest_id) as prod_dest,
# MAGIC     COUNT(*) AS cantidad_compras
# MAGIC FROM tab_min
# MAGIC GROUP BY cliente_id
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tm

# COMMAND ----------

df = spark.sql(""" 
                select * 
                from 
                    tm
                """
                )

# COMMAND ----------

df = df.drop('cliente_id')

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
prod_dest_indexer = StringIndexer(inputCol='prod_dest', outputCol= 'prod_destIndex')
prod_dest_encoder = OneHotEncoder(inputCol = 'prod_destIndex', outputCol= 'prod_destVec')

# COMMAND ----------

Nombre_indexer = StringIndexer(inputCol='Nombre', outputCol= 'NombreIndex')
Nombre_encoder = OneHotEncoder(inputCol = 'NombreIndex', outputCol= 'NombreVec')

# COMMAND ----------

# 'stateVec',
assembler = VectorAssembler(inputCols = [ 'prod_destVec','NombreVec',
                                        'precio_total',
                                         'prod_vendido',
                                         'stock',
                                         'prod_mov',
                                         'cantidad_compras'
                                        ], outputCol= 'features')

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
kmeans = KMeans().setK(5).setSeed(1)


# COMMAND ----------

pipeline = Pipeline(stages= [
    # state_indexer,
    prod_dest_indexer,
    Nombre_indexer,
    # state_encoder,
    prod_dest_encoder,
    Nombre_encoder,
    assembler, 
    kmeans])

# COMMAND ----------

train_data, test_data = df.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------


