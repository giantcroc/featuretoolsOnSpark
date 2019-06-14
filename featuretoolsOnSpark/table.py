# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import pandas as pd
import time
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number,col

import featuretoolsOnSpark.column_types as ctypes

import re

from featuretoolsOnSpark.util import *

class Table(object):
    """Represents a table in a tableset, and stores relevant metadata and data

    Attributes:
        id
        tableset
        verbose
        df
        index
        num_df

    """
    def __init__(self, id, df,tableset,num_df=None, column_types=None,
                 index=None, make_index=False,verbose=True):
        """ Create Table

        Args:
            id (str): Id of Table.

            df (pyspark.sql.DataFrame): Dataframe providing the data for the Table.

            num_df (int, optional): How many rows of pyspark.sql.DataFrame which are converted to pd.DataFrame to ensure type of columns.

            tableset (TableSet): Tableset for this Table.

            column_types (dict[str -> dict[str -> type]]) : A table's column_types dict maps string column ids to types (:class:`.Column`)
                or (type, kwargs) to pass keyword arguments to the Column.

            index (str): Name of id column in the dataframe.

            make_index (bool, optional) : If True, assume index does not exist as a column in
                dataframe, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in dataframe.

            verbose (bool) : Whether to display information
        """
        cnt = df.count()
        if num_df is not None:
            assert num_df>0,"num_df must be greater than 0"
            assert num_df<=cnt,"num_df must be less than row num of df"
        else:
            assert cnt>10,"the numbers of dataframe must be greater than 10"
            num_df = 10

        self._validate_table_params(id, df)

        self.id = id
        self.tableset = tableset
        self.pdf = df.limit(num_df).toPandas()
        self.verbose = verbose
        self.df = df
        self.num_df = num_df
        self.cnt = cnt
        self.old_len = len(df.columns)

        self._create_index(index, make_index)

        self._create_columns(column_types)

        self.interesting_columns=[]

        if self.verbose:
            logger.info("create table "+self.id)
    
    def _create_columns(self, column_types):
        """Extracts the columns from a dataframe

        Args:
            column_types (dict[str -> dict[str -> type]]) : A table's
                column_types dict maps string column ids to types (:class:`.Column`)
                or (type, kwargs) to pass keyword arguments to the Column.
        """
        columns = []
        column_types = column_types or {}
        if self.index not in column_types:
            column_types[self.index] = ctypes.Index

        link_cols = self.get_linked_cols()
        
        inferred_column_types = self.infer_column_types(link_cols,column_types)

        inferred_column_types.update(column_types)

        for c in inferred_column_types:
            ctype = inferred_column_types[c]
            if isinstance(ctype, tuple):
                _c = ctype[0](c, self, ctype[1])
            else:
                _c = inferred_column_types[c](c, self)
            columns += [_c]
        # make sure index is at the beginning
        index_column = [c for c in columns if c.id == self.index][0]
        self.columns = [index_column] + [c for c in columns if c.id != self.index]

    def get_linked_cols(self):
        """Return a list with the table linked columns.
        """
        link_relationships = [r for r in self.tableset.relationships
                            if r.parent_table.id == self.id or
                            r.child_table.id == self.id]
        link_cols = [v.id for rel in link_relationships
                    for v in [rel.parent_column, rel.child_column]
                    if v.table.id == self.id]
        return link_cols

    def infer_column_types(self, link_cols, column_types):
        '''Infer column types from dataframe

        Args:
            link_cols (list[str]): Linked columns
            column_types (dict[str -> dict[str -> type]]) : A table's column_types dict maps string column ids 
                to types (:class:`.Column`)
                or (type, kwargs) to pass keyword arguments to the column.
        '''  
        df = self.pdf

        inferred_types = {}
        for column in df.columns:
            column = column.encode("utf-8")
            inferred_type = None
            if column in column_types:
                continue
            else:
                if len(df[column].dropna())==0:
                    col = self.df.select(column).dropna().limit(self.num_df).toPandas()[column]
                else:
                    col = df[column]
                col = col.dropna().reset_index(drop=True)

                if len(col)==0:
                    continue
                elif col.dtype.name == "object":
                    if column in link_cols:
                        inferred_type = ctypes.Categorical
                    else:
                        is_datetime,formatt = self.col_is_datetime(col)
                        if is_datetime:
                            inferred_type = (ctypes.Datetime,formatt)
                        else:
                            inferred_type = ctypes.Categorical

                elif col.dtype.name == "bool":
                    inferred_type = ctypes.Boolean
                elif column in link_cols:
                    inferred_type = ctypes.Ordinal
                else:
                    is_datetime,formatt = self.col_is_datetime(col)
                    if is_datetime:
                        inferred_type = (ctypes.Datetime,formatt)
                    else:
                        inferred_type = ctypes.Numeric

            if inferred_type != None:
                inferred_types[column] = inferred_type

        return inferred_types

    def col_is_datetime(self,col):
        if col.dtype.name.find('datetime') > -1:
            return True,col.dtype.name
        def is_time_format(data):
            if data[:4] < '1970' or data[:4] > '2020':
                return False
            if data[4:6] > '12' or data[4:6] < '01':
                return False
            if data[6:8] > '31' or data[6:8] < '01':
                return False
            return True
        if col.dtype.name.find('int32') > -1 or col.dtype.name.find('int64') > -1 or col.dtype.name.find('int16') > -1:
            col = col.astype(str)
        # re match two patterns:1.2013[-/]04[-/]29 2.2013[-/]04[-/]29 03:04:30
        if col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
            
            pattern1 = r"\d{4}[-/]?\d{2}[-/]?\d{2}"
            pattern2 = r"\d{4}[-/]?d{2}[-/]?d{2}\s\d{2}:\d{2}:\d{2}"
            if re.match(pattern1,col[0]):
                if is_time_format(col[0]):
                    return True,'year[-/]month[-/]day'
            if re.match(pattern2,col[0]):
                return True,'year[-/]month[-/]day hour:min:sec'
        return False,None

    def __getitem__(self, column_id):
        return self._get_column(column_id)

    def _get_column(self, column_id):
        """Get column instance

        Args:
            column_id (str) : Id of column to get.

        Returns:
            :class:`.Column` : Instance of column.

        Raises:
            RuntimeError : if no column exist with provided id
        """
        for v in self.columns:
            if v.id == column_id:
                return v

        raise KeyError("Column: %s not found in table" % (column_id))

    def _get_column_ids(self):
        """Get column ids

        Args:
            column_id (str) : Id of column to get.

        Returns:
            :[str]: ids of column.
        """
        return [column.id for column in self.columns]

    def _get_column_index(self, column_id):
        """Get column index in self.columns

        Args:
            column_id (str) : Id of column to get.

        Returns:
            :int:index of column.

        Raises:
            RuntimeError : if no column exist with provided id
        """
        for i,v in enumerate(self.columns):
            if v.id == column_id:
                return i

        raise KeyError("Column: %s not found in table" % (column_id))


    def convert_column_type(self, column_id, new_type,
                              **kwargs):
        """Convert column in dataframe to different type

        Args:
            column_id (str) : Id of column to convert.
            new_type (subclass of `Column`) : Type of column to convert to.
        """
        # replace the old column with the new one, maintaining order
        column = self._get_column(column_id)
        new_column = new_type.create_from(column)
        self.columns[self._get_column_index(column_id)] = new_column

    def convert_column_id(self,column_id,new_id):
        
        self.df = self.df.withColumnRenamed(column_id,new_id)

        index = self._get_column_index(column_id)

        column = self._get_column(column_id)
        column.id = new_id
        self.columns[index] = column

    def __repr__(self):
        repr_out = u"Table: {}\n".format(self.id)
        repr_out += u"  Columns:"
        for v in self.columns:
            repr_out += u"\n    {} (dtype: {})".format(v.id, v.type_string)

        shape = self.shape
        repr_out += u"\n  Shape:\n    (Rows: {}, Columns: {})".format(
            shape[0], shape[1])

        # encode for python 2
        if type(repr_out) != str:
            repr_out = repr_out.encode("utf-8")

        return repr_out

    @property
    def shape(self):
        '''Shape of the entity's dataframe'''
        return (self.cnt,len(self.columns))

    def _validate_table_params(self,id,df):
        '''Validation checks for Table inputs'''
        assert isinstance(id,str), "Table id must be a string"
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
        for c in df.columns:
            if not isinstance(c,str):
                raise ValueError("All column names must be strings (Column {} "
                                "is not a string)".format(c))

    def _create_index(self,index, make_index):
        '''Handles index creation logic base on user input'''


        if index is None:
            # Case 1: user wanted to make index but did not specify column name
            assert not make_index, "Must specify an index name if make_index is True"
            # Case 2: make_index not specified but no index supplied, use first column
            logger.warning(("Using first column as index. ",
                            "To change this, specify the index parameter"))
            index = self.df.columns[0]
        elif make_index and index in self.df.columns:
            # Case 3: user wanted to make index but column already exists
            raise RuntimeError("Cannot make index: index column already present")
        elif index not in self.df.columns:
            if not make_index:
                # Case 4: user names index, it is not in df. does not specify
                # make_index.  Make new index column and warn
                logger.warning("index %s not found in dataframe, creating new "
                            "integer column", index)
            # Case 5: make_index with no errors or warnings
            # (Case 4 also uses this code path)

            rank_window = Window().orderBy(col(self.df.columns[0]))
            new_add_col = row_number().over(rank_window)

            self.df = self.df.withColumn(index,new_add_col-1)

        # Case 6: user specified index, which is already in df. No action needed.
        self.index=index
