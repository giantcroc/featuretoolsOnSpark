from __future__ import division

import numpy as np
import pandas as pd

class Column(object):
    """Represent a Column in an table

    Args:
        id (str) : Id of column. The name in table.
        table (:class:`.Table`) : Table this column belongs to.
        name (str, optional) : column name. Defaults to id.

    
    """
    type_string = None
    _default_dtype = object

    def __init__(self, id, table, name=None):
        assert isinstance(id,str), "Column id must be a string"
        self.id = id
        self._name = name
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
        ret = u"<Column: {} (dtype = {})>".format(self.name, self.type_string)

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
        v = cls(id=column.id, name=column.name, table=column.table)
        return v

    @property
    def name(self):
        return self._name if self._name is not None else self.id

    @property
    def dtype(self):
        return self.type_string \
            if self.type_string is not None else "generic_type"

    @name.setter
    def name(self, name):
        self._name = name

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
                'name': self.name,
                'table': self.table.id,
                'interesting_values': self._interesting_values
            },
        }

if __name__ == "__main__":
    test=Column("111",1)
    test1=Column("111",2)
    print(test==test1)