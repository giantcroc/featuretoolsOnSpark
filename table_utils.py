# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

import column_types as ctypes

def get_linked_cols(table):
    """Return a list with the table linked columns.
    """
    link_relationships = [r for r in table.tableset.relationships
                          if r.parent_table.id == table.id or
                          r.child_table.id == table.id]
    link_cols = [v.id for rel in link_relationships
                 for v in [rel.parent_column, rel.child_column]
                 if v.table.id == table.id]
    return link_cols

def infer_column_types(df, link_cols, column_types):
    '''Infer column types from dataframe

    Args:
        df (DataFrame): Input DataFrame
        link_cols (list[]): Linked columns
        column_types (dict[str -> dict[str -> type]]) : A table's column_types dict maps string column ids 
            to types (:class:`.Column`)
            or (type, kwargs) to pass keyword arguments to the column.
    '''
    inferred_types = {}
    inferred_type = ctypes.Unknown
    for column in df.columns:
        if column in column_types:
            continue
        elif df[column].dtype == "object":
            if column in link_cols:
                inferred_type = ctypes.Categorical
            elif len(df[column].dropna()):
                if col_is_datetime(df[column]):
                    inferred_type = ctypes.Datetime
                else:
                    inferred_type = ctypes.Categorical

        elif df[column].dtype == "bool":
            inferred_type = ctypes.Boolean

        elif col_is_datetime(df[column]):
            inferred_type = ctypes.Datetime

        elif column in link_cols:
            inferred_type = ctypes.Ordinal

        elif len(df[column].dropna()):
            inferred_type = ctypes.Numeric

        inferred_types[column] = inferred_type

    return inferred_types

def col_is_datetime(col):
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.iloc[0], datetime))):
        return True

    # TODO: not sure this is ideal behavior.
    # it converts int columns that have dtype=object to datetimes starting from 1970
    elif col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
        try:
            pd.to_datetime(col.dropna().iloc[:5], errors='raise')
        except Exception:
            return False
        else:
            return True
    return False