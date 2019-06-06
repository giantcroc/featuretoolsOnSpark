import logging
import numpy as np
import pandas as pd

import column_types as ctypes
from table import Table
from relationship import Relationship

logging.basicConfig(format = '%(module)s-%(levelname)s- %(message)s')
logger = logging.getLogger('featuretoolsOnSpark')
logger.setLevel(20)

class TableSet(object):
    """
    Stores all actual data for a tableset

    Attributes:
        id
        table_dict
        relationships
        no_change_columns

    """
    def __init__(self, id=None, tables=None, relationships=None, no_change_columns=None):
        """Creates TableSet

            Args:
                id (str) : Unique identifier to associate with this instance

                tables (dict[str -> tuple(pyspark.sql.DataFrame, str, str)]): Dictionary of
                    tables. Entries take the format
                    {table id -> (dataframe, id column, (column_types))}.
                    Note that column_types are optional.

                relationships (list[(str, str, str, str)]): List of relationships
                    between tables. List items are a tuple with the format
                    (parent table id, parent column, child table id, child column).

                no_change_columns([str]): ids of the columns that can't be changed even if there are duplication of id.
        """
        self.id = id or "tableset"
        self.table_dict = {}
        self.relationships = []

        logger.info("create tableset "+self.id)

        self.no_change_columns = no_change_columns or []
        tables = tables or {}
        relationships = relationships or []
        for table in tables:
            df = tables[table][0]
            index_column = tables[table][1]
            column_types = None
            if len(tables[table]) > 2:
                column_types = tables[table][2]
            self.table_from_dataframe(table_id=table,
                                       dataframe=df,
                                       index=index_column,
                                       column_types=column_types)

        for relationship in relationships:
            parent_column = self[relationship[0]][relationship[1]]
            child_column = self[relationship[2]][relationship[3]]
            self.add_relationship(Relationship(parent_column,
                                               child_column))
    
    def table_from_dataframe(self,
                              table_id,
                              dataframe,
                              index=None,
                              num_df=None,
                              column_types=None,
                              make_index=False):
        """
        Load the data for a specified table from a Spark DataFrame.

        Args:
            table_id (str) : Unique id to associate with this table.

            dataframe (pyspark.sql.DataFrame) : Dataframe containing the data.

            index (str, optional): Name of the column used to index the table.
                If None, take the first column.

            column_types (dict[str -> Variable], optional):
                Keys are of column ids and values are column types. Used to to
                initialize a table's store.

            make_index (bool, optional) : If True, assume index does not
                exist as a column in dataframe, and create a new column of that name
                using integers. Otherwise, assume index exists.

            num_df (int, optional): How many rows of pyspark.sql.DataFrame which are converted to pd.DataFrame.
                Needed when data is the format of pyspark.sql.DataFrame.

        Notes:

            Will infer column types from Pandas dtype
        """
        column_types = column_types or {}
        table = Table(
            table_id,
            dataframe,
            self,
            num_df=num_df,
            column_types=column_types,
            index=index,
            make_index=make_index)

        for k,v in self.table_dict.items():
            inters = set(v._get_column_ids()).intersection(set(table._get_column_ids()))
            if len(inters)>0:

                for inter in inters:
                    if inter not in self.no_change_columns:
                        v.convert_column_id(inter,k+'_'+inter)
                        table.convert_column_id(inter,table.id+'_'+inter)

        self.table_dict[table.id] = table

        return self
    

    @property
    def tables(self):
        return list(self.table_dict.values())

    def __getitem__(self, table_id):
        """Get table instance from tableset

        Args:
            table_id (str): Id of table.

        Returns:
            :class:`.Table` : Instance of table. None if table doesn't
                exist.
        """
        if table_id in self.table_dict:
            return self.table_dict[table_id]
        name = self.id or "table set"
        raise KeyError('Table %s does not exist in %s' % (table_id, name))

    def __repr__(self):
        repr_out = u"Tableset: {}\n".format(self.id)
        repr_out += u"  Tables:"
        for e in self.tables:
            if e.df.shape:
                repr_out += u"\n    {} [Rows: {}, Columns: {}]".format(
                    e.id, e.raw_data.count(), e.df.shape[1])
            else:
                repr_out += u"\n    {} [Rows: None, Columns: None]".format(
                    e.id)
        repr_out += "\n  Relationships:"

        if len(self.relationships) == 0:
            repr_out += u"\n    No relationships"

        for r in self.relationships:
            repr_out += u"\n    %s.%s -> %s.%s" % \
                (r._child_table_id, r._child_column_id,
                 r._parent_table_id, r._parent_column_id)

        # encode for python 2
        if type(repr_out) != str:
            repr_out = repr_out.encode("utf-8")

        return repr_out

    def add_relationships(self, relationships):
        """Add multiple new relationships to a tableset

        Args:
            relationships (list[Relationship]) : List of new
                relationships.
        """
        return [self.add_relationship(r) for r in relationships][-1]

    def add_relationship(self, relationship):
        """Add a new relationship between tables in the tableset

        Args:
            relationship (Relationship) : Instance of new
                relationship to be added.
        """
        if relationship in self.relationships:
            logger.warning(
                "Not adding duplicate relationship: %s", relationship)
            return self

        # this is a new pair of tables
        child_e = relationship.child_table
        child_v = relationship.child_column.id
        parent_e = relationship.parent_table
        parent_v = relationship.parent_column.id
        if not isinstance(child_e[child_v], ctypes.Id):
            child_e.convert_column_type(column_id=child_v,
                                          new_type=ctypes.Id)

        if not isinstance(parent_e[parent_v], ctypes.Index):
            parent_e.convert_column_type(column_id=parent_v,
                                           new_type=ctypes.Index)

        self.relationships.append(relationship)
        return self

    def get_backward_tables(self, table_id, deep=False):
        """Get tables that are in a backward relationship with table

        Args:
            table_id (str) - Id table of table to search from.

        Returns:
            Set of each :class:`.Table` in a backward relationship.
        """
        return set([r.child_table.id for r in
                    self.get_backward_relationships(table_id)])

    def get_backward_relationships(self, table_id):
        """
        get relationships where table "table_id" is the parent.

        Args:
            table_id (str): Id of table to get relationships for.

        Returns:
            list[:class:`.Relationship`]: list of backward relationships
        """
        return [r for r in self.relationships if r.parent_table.id == table_id]

    def get_forward_tables(self, table_id, deep=False):
        """Get tables that are in a forward relationship with table

        Args:
            table_id (str) - Id table of table to search from.

        Returns:
            Set of table IDs in a forward relationship with the passed in
            table.
        """
        return set([r.parent_table.id for r in
                   self.get_forward_relationships(table_id)])


    def get_forward_relationships(self, table_id):
        """Get relationships where table "table_id" is the child

        Args:
            table_id (str): Id of table to get relationships for.

        Returns:
            list[:class:`.Relationship`]: List of forward relationships.
        """
        return [r for r in self.relationships if r.child_table.id == table_id]

    def find_backward_path(self, start_table_id, goal_table_id):
        """Find a backward path between a start and goal table

        Args:
            start_table_id (str) : Id of table to start the search from.
            goal_table_id  (str) : Id of table to find backward path to.

        Returns:
            List of relationship that go from start table to goal table. None
            is returned if no path exists.
        """
        forward_path = self.find_forward_path(goal_table_id, start_table_id)
        if forward_path is not None:
            return forward_path[::-1]
        return None

    def find_forward_path(self, start_table_id, goal_table_id):
        """Find a forward path between a start and goal table

        Args:
            start_table_id (str) : id of table to start the search from
            goal_table_id  (str) : if of table to find forward path to

        Returns:
            List of relationships that go from start table to goal
                table. None is return if no path exists

        """

        if start_table_id == goal_table_id:
            return []

        for r in self.get_forward_relationships(start_table_id):
            new_path = self.find_forward_path(
                r.parent_table.id, goal_table_id)
            if new_path is not None:
                return [r] + new_path

        return None
if __name__ == "__main__":
    tableset=TableSet("id")