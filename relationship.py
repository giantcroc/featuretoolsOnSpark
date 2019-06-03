class Relationship(object):
    """Class to represent an relationship between tables

    See Also:
        :class:`.TableSet`, :class:`.Table`, :class:`.Column`
    """

    def __init__(self, parent_column, child_column):
        """ Create a relationship

        Args:
            parent_column (:class:`.Discrete`): Instance of column
                in parent table.  Must be a Discrete Column
            child_column (:class:`.Discrete`): Instance of column in
                child table.  Must be a Discrete Column

        """

        self.tableset = child_column.tableset
        self._parent_table_id = parent_column.table.id
        self._child_table_id = child_column.table.id
        self._parent_column_id = parent_column.id
        self._child_column_id = child_column.id

        if (parent_column.table.index is not None and parent_column.id != parent_column.table.index):
            raise AttributeError("Parent column '%s' is not the index of table %s" % (parent_column, parent_column.table))

    def __repr__(self):
        ret = u"<Relationship: %s.%s -> %s.%s>" % \
            (self._child_table_id, self._child_column_id,
             self._parent_table_id, self._parent_column_id)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._parent_table_id == other._parent_table_id and \
            self._child_table_id == other._child_table_id and \
            self._parent_column_id == other._parent_column_id and \
            self._child_column_id == other._child_column_id

    @property
    def parent_table(self):
        """Parent table object"""
        return self.tableset[self._parent_table_id]

    @property
    def child_table(self):
        """Child table object"""
        return self.tableset[self._child_table_id]

    @property
    def parent_column(self):
        """Instance of column in parent table"""
        return self.parent_table[self._parent_column_id]

    @property
    def child_column(self):
        """Instance of column in child table"""
        return self.child_table[self._child_column_id]
