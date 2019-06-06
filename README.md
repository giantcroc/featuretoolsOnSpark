# featuretoolsOnSpark
[Featuretools](https://github.com/Featuretools/featuretools) is a python library for automated feature engineering.

This repo is a simplified version of featuretools,using  automatic feature generation framework of featuretools.Instead of the fussy back-end architecture of featuretools,We mainly use [Spark DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame) to achieve faster feature generation process(speed up 10x+).
