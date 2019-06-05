from __future__ import division

import numpy as np
import pandas as pd

class Column(object):
    """Represent a Column in a table

    Args:
        id (str) : Id of column. The name in table.
        table (:class:`.Table`) : Table this column belongs to.
    
    """
    type_string = None
    _default_dtype = object

    def __init__(self, id, table):
        assert isinstance(id,str), "Column id must be a string"
        self.id = id
        self.table_id = table.id
        assert table.tableset is not None, "table must contain reference to TableSet"
        self.table = table
        self._interesting_values = None

    @property
    def tableset(self):
        return self.table.tableset

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.id == other.id and self.table_id == other.table_id
    
    def __repr__(self):
        ret = u"<Column: {} (dtype = {})>".format(self.id, self.type_string)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    @classmethod
    def create_from(cls, column):
        """Create new column this type from existing

        Args:
            column (Column) : Existing column to create from.

        Returns:
            :class:`.Column` : new column

        """
        v = cls(id=column.id, table=column.table)
        return v

    @property
    def dtype(self):
        return self.type_string \
            if self.type_string is not None else "generic_type"

    @property
    def interesting_values(self):
        return self._interesting_values

    @interesting_values.setter
    def interesting_values(self, interesting_values):
        self._interesting_values = interesting_values

    @property
    def series(self):
        return self.table.df[self.id]

    def to_data_description(self):
        return {
            'id': self.id,
            'type': {
                'value': self.type_string,
            },
            'properties': {
                'table': self.table.id,
                'interesting_values': self._interesting_values
            },
        }

class Unknown(Column):
    pass


class Discrete(Column):
    """Superclass representing columns that take on discrete values"""
    type_string = "discrete"

    def __init__(self, id, table):
        super(Discrete, self).__init__(id, table)
        self._interesting_values = []

    @property
    def interesting_values(self):
        return self._interesting_values

    @interesting_values.setter
    def interesting_values(self, values):
        seen = set()
        seen_add = seen.add
        self._interesting_values = [v for v in values
                                    if not (v in seen or seen_add(v))]

class Boolean(Column):
    """Represents Columns that take on one of two values

    Args:
        true_values (list) : List of valued true values. Defaults to [1, True, "true", "True", "yes", "t", "T"]
        false_values (list): List of valued false values. Defaults to [0, False, "false", "False", "no", "f", "F"]
    """
    type_string = "boolean"
    _default_pandas_dtype = bool

    def __init__(self,
                 id,
                 table,
                 true_values=None,
                 false_values=None):
        default = [1, True, "true", "True", "yes", "t", "T"]
        self.true_values = true_values or default
        default = [0, False, "false", "False", "no", "f", "F"]
        self.false_values = false_values or default
        super(Boolean, self).__init__(id, table)

    def to_data_description(self):
        description = super(Boolean, self).to_data_description()
        description['type'].update({
            'true_values': self.true_values,
            'false_values': self.false_values
        })
        return description


class Categorical(Discrete):
    """Represents Columns that can take an unordered discrete values

    Args:
        categories (list) : List of categories. If left blank, inferred from data.
    """
    type_string = "categorical"

    def __init__(self, id, table, categories=None):
        self.categories = None or []
        super(Categorical, self).__init__(id, table)

    def to_data_description(self):
        description = super(Categorical, self).to_data_description()
        description['type'].update({'categories': self.categories})
        return description


class Id(Categorical):
    """Represents Columns that identify another table"""
    type_string = "id"
    _default_pandas_dtype = int


class Ordinal(Discrete):
    """Represents Columns that take on an ordered discrete value"""
    type_string = "ordinal"
    _default_pandas_dtype = int


class Numeric(Column):
    """Represents Columns that contain numeric values

    Args:
        range (list, optional) : List of start and end. Can use inf and -inf to represent infinity. Unconstrained if not specified.
        start_inclusive (bool, optional) : Whether or not range includes the start value.
        end_inclusive (bool, optional) : Whether or not range includes the end value

    Attributes:
        max (float)
        min (float)
        std (float)
        mean (float)
    """
    type_string = "numeric"
    _default_pandas_dtype = float

    def __init__(self,
                 id,
                 table,
                 range=None,
                 start_inclusive=True,
                 end_inclusive=False):
        self.range = None or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        super(Numeric, self).__init__(id, table)

    def to_data_description(self):
        description = super(Numeric, self).to_data_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive,
        })
        return description


class Index(Column):
    """Represents Columns that uniquely identify an instance of an table

    Attributes:
        count (int)
    """
    type_string = "index"
    _default_pandas_dtype = int


class Datetime(Column):
    """Represents Columns that are points in time

    Args:
        format (str): Python datetime format string documented `here <http://strftime.org/>`_.
    """
    type_string = "datetime"
    _default_pandas_dtype = np.datetime64

    def __init__(self, id, table, format=None):
        self.format = format
        super(Datetime, self).__init__(id, table)

    def __repr__(self):
        ret = u"<Column: {} (dtype: {}, format: {})>".format(self.id, self.type_string, self.format)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    def to_data_description(self):
        description = super(Datetime, self).to_data_description()
        description['type'].update({'format': self.format})
        return description


class TimeIndex(Column):
    """Represents time index of table"""
    type_string = "time_index"
    _default_pandas_dtype = np.datetime64


class NumericTimeIndex(TimeIndex, Numeric):
    """Represents time index of table that is numeric"""
    type_string = "numeric_time_index"
    _default_pandas_dtype = float


class DatetimeTimeIndex(TimeIndex, Datetime):
    """Represents time index of table that is a datetime"""
    type_string = "datetime_time_index"
    _default_pandas_dtype = np.datetime64


class Timedelta(Column):
    """Represents Columns that are timedeltas

    Args:
        range (list, optional) : List of start and end of allowed range in seconds. Can use inf and -inf to represent infinity. Unconstrained if not specified.
        start_inclusive (bool, optional) : Whether or not range includes the start value.
        end_inclusive (bool, optional) : Whether or not range includes the end value
    """
    type_string = "timedelta"
    _default_pandas_dtype = np.timedelta64

    def __init__(self,
                 id,
                 table,
                 range=None,
                 start_inclusive=True,
                 end_inclusive=False):
        self.range = range or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        super(Timedelta, self).__init__(id, table)

    def to_data_description(self):
        description = super(Timedelta, self).to_data_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive,
        })
        return description


class Text(Column):
    """Represents Columns that are arbitary strings"""
    type_string = "text"
    _default_pandas_dtype = str


class PandasTypes(object):
    _all = 'all'
    _categorical = 'category'
    _pandas_datetimes = ['datetime64[ns]', 'datetime64[ns, tz]']
    _pandas_timedeltas = ['Timedelta']
    _pandas_numerics = ['int16', 'int32', 'int64',
                        'float16', 'float32', 'float64']


DEFAULT_DTYPE_VALUES = {
    np.datetime64: pd.Timestamp.now(),
    int: 0,
    float: 0.1,
    np.timedelta64: pd.Timedelta('1d'),
    object: 'object',
    bool: True,
    str: 'test'
}


if __name__ == "__main__":
    test=Column("111",1)
    test1=Column("111",2)
    print(test==test1)