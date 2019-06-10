# featuretoolsOnSpark
[Featuretools](https://github.com/Featuretools/featuretools) is a python library for automated feature engineering.

This repo is a simplified version of featuretools,using  automatic feature generation framework of featuretools.Instead of the fussy back-end architecture of featuretools,We mainly use [Spark DataFrame](http://spark.apache.org/docs/latest/api/python/index.html#) to achieve faster feature generation process(speed up 10x+).

## Installation
Install with pip

	pip install featuretoolsOnSpark
Install from source

	git clone https://github.com/giantcroc/featuretoolsOnSpark.git
	cd featuretoolsOnSpark
	python setup.py install
	
## Example
Below is an example of how to use apis of this repo.We Choose the dataset from Kaggle's competition([Home-Credit-Default-Risk](https://www.kaggle.com/c/home-credit-default-risk/data)).The relationships between tables are shown in the picture below.

<p align="center">
<img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png" alt="featuretoolsOnSpark" />
</p>

First,you should guarantee that all csv files needed have been saved as [Spark DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame) format.

#### 1. Create Spark Context
```python
>> from pyspark.sql import SparkSession

>> spark = SparkSession \
   	.builder \
   	.appName("home-credit") \
   	.enableHiveSupport()\
   	.getOrCreate()
```
#### 2. Get Spark DataFrame
```python
>> app_train = spark.sql(''' select * from home_credit.app_train ''')

>> bureau = spark.sql(''' select * from home_credit.bureau ''')

>> bureau_balance = spark.sql(''' select * from home_credit.bureau_balance ''')

>> cash = spark.sql(''' select * from home_credit.cash ''')

>> credit = spark.sql(''' select * from home_credit.credit ''')

>> installments = spark.sql(''' select * from home_credit.installments ''')

>> previous = spark.sql(''' select * from home_credit.previous ''')
```
#### 3. Create TableSet
```python
>> import featuretoolsOnSpark as fts

>> ts = fts.TableSet("home_credit",no_change_columns=["SK_ID_PREV","SK_ID_CURR","SK_ID_BUREAU"])
```
#### 4. Create Tables From Spark DataFrame
```python
>> ts.table_from_dataframe(table_id="bureau_balance",dataframe=bureau_balance,index='bureau_balance_id',make_index = True)

>> ts.table_from_dataframe(table_id="app_train",dataframe=app_train,index='SK_ID_CURR')

>> ts.table_from_dataframe(table_id="bureau",dataframe=bureau,index='SK_ID_BUREAU')

>> ts.table_from_dataframe(table_id="cash",dataframe=cash,index='cash_id',make_index = True)

>> ts.table_from_dataframe(table_id="credit",dataframe=credit,index='credit_id',make_index = True)

>> ts.table_from_dataframe(table_id="installments",dataframe=installments,index='installments_id',make_index = True)

>> ts.table_from_dataframe(table_id="previous",dataframe=previous,index='SK_ID_PREV')
```
#### 5. Add Relationships of Tables
```python
>> re1 = Relationship(ts["app_train"]["SK_ID_CURR"],ts["bureau"]["SK_ID_CURR"])

>> re2 = Relationship(ts["bureau"]["SK_ID_BUREAU"],ts["bureau_balance"]["SK_ID_BUREAU"])

>> re3 = Relationship(ts["app_train"]["SK_ID_CURR"],ts["previous"]["SK_ID_CURR"])

>> re4 = Relationship(ts["previous"]["SK_ID_PREV"],ts["cash"]["SK_ID_PREV"])

>> re5 = Relationship(ts["previous"]["SK_ID_PREV"],ts["credit"]["SK_ID_PREV"])

>> re6 = Relationship(ts["previous"]["SK_ID_PREV"],ts["installments"]["SK_ID_PREV"])

>> ts.add_relationships([re1,re2,re3,re4,re5,re6])
```
#### 6. Run DFS To Generate Features
```python
new_app_train = fts.dfs(tableset = ts, agg_primitives=["sum",'min','max','avg'],target_table = 'app_train',max_depth=2)
```
