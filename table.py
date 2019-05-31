# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from pyspark.sql import DataFrame

#import column_types as ctypes
import logging

logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('featureflow')

class Table(object):
    """Represents an table in a tableset, and stores relevant metadata and data

    """
    def __init__(self, id, data,tableset,num_df=None, columns_types=None,
                 index=None, make_index=False,verbose=False):
        """ Create Table

        Args:
            id (str): Id of Table.
            data (pyspark.sql.DataFrame): Dataframe providing the data for the Table.
            num_df (int, optional): How many rows of pyspark.sql.DataFrame which are converted to pd.DataFrame.
                Needed when data is the format of pyspark.sql.DataFrame.
            entityset (EntitySet): Entityset for this Entity.
            column_types (dict[str -> dict[str -> type]]) : An table's column_types dict maps string column ids to types (:class:`.Column`)
                or (type, kwargs) to pass keyword arguments to the Column.
            index (str): Name of id column in the dataframe.
            make_index (bool, optional) : If True, assume index does not exist as a column in
                dataframe, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in dataframe.
        """
        if num_df:
            assert num_df>0,"num_df must be greater than 0"
        else:
            cnt = data.count()
            assert cnt>10,"the numbers of dataframe must be greater than 10"
            num_df = int(cnt/10)
        df = data.limit(num_df).toPandas()
        print(df)
        _validate_table_params(id, df)
        #created_index, index, df = _create_index(index, make_index, df)

def _validate_table_params(id, df):
    '''Validation checks for Table inputs'''
    assert isinstance(id,str), "Table id must be a string"
    assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
    for c in df.columns:
        if not isinstance(c,str):
            raise ValueError("All column names must be strings (Column {} "
                             "is not a string)".format(c))

if __name__ == "__main__":
    logger.warning("ssssssssss")
    print("sss")