# Databricks notebook source
DROP TABLE IF EXISTS account;

create table account
(account_id float,
district_id float,
frequency string,
`date` float)
USING csv
OPTIONS (path "/FileStore/data/account.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

DROP TABLE IF EXISTS client;
create table client
USING csv
OPTIONS (path "/FileStore/data/client.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

DROP TABLE IF EXISTS disp;
create table disp
USING csv
OPTIONS (path "/FileStore/data/disp.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

DROP TABLE IF EXISTS district;
create table district
USING csv
OPTIONS (path "/FileStore/data/district.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

DROP TABLE IF EXISTS loan;
create table loan
USING csv
OPTIONS (path "/FileStore/data/loan.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

CREATE OR REPLACE view tabla_minable 
as
SELECT *
FROM 
(
	SELECT client_id, district_id
	     , if (SUBSTRING(birth_number, 3, 2) <= '50', 'hombre', 'mujer') as genero -- extrayendo el genero
    FROM 
	    client
) as c
   LEFT JOIN (
				SELECT A1 as districtid
					 , A2 as distrito
					 , A3 as region
					 , A4 as habitantes
					 , A9 as num_ciudades
					 , A10 as ratio_residentes_urbanos
					 , A11 as Salario_promedio
				FROM 
					district
             ) as dis on dis.districtid = c.district_id
  INNER JOIN (
				SELECT client_id as c_id , account_id, type as tipo_disposicion_cliente
                FROM
					disp
			 ) as di on di.c_id = c.client_id
  INNER JOIN (
				SELECT account_id as ac_id
					 -- , district_id as district_id_cuenta -- info  a consideracion debido a que nos interesa el distrito del cliente
					 -- , STR_TO_DATE(date, '%y%m%d') as fecha_creacion_cuenta -- conversion de la fecha
					 , case
						  when frequency = 'POPLATEK MESICNE' then 'emision mensual'
						  when frequency = 'POPLATEK TYDNE' then 'emision semanal'
						  when frequency = 'POPLATEK PO OBRATU' then 'emision despues de una transaccion'
					    else 'no especifica'
					    end as frecuencia_emision			
				FROM 
					account
			 ) as ac on ac.ac_id = di.account_id

 LEFT JOIN (
				SELECT loan_id, account_id as accl_id
					 , amount as monto_prestamo, duration as duracion_prestamo
					 , payments AS pagos_mensuales
                     , status  as estado
				FROM 
					loan
             ) as ln on ln.accl_id = ac.ac_id

;

# COMMAND ----------


