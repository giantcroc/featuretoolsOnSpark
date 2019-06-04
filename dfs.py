import pandas as pd

from tableset import TableSet
from collections import defaultdict
import column_types as ctypes
from pyspark.sql import SparkSession

def dfs(tables=None,
        relationships=None,
        tableset=None,
        target_table=None,
        agg_primitives=None,
        allowed_paths=None,
        max_depth=2,
        ignore_tables=None,
        ignore_columns=None,
        where_primitives=None,
        max_features=-1,
        save_progress=None,
        features_only=False,
        verbose=False,
        return_column_types=None,
        context=None):
    '''Calculates a feature matrix and features given a dictionary of tables
    and a list of relationships.


    Args:
        tables (dict[str -> tuple(pyspark.sql.DataFrame, str, str)]): Dictionary of
            tables. Entries take the format
            {table id -> (dataframe, id column)}.

        relationships (list[(str, str, str, str)]): List of relationships
            between tables. List items are a tuple with the format
            (parent table id, parent column, child table id, child column).

        tableset (TableSet): An already initialized tableset. Required if
            tables and relationships are not defined.

        target_table (str): Table id of table on which to make predictions.

        agg_primitives (list[str or AggregationPrimitive], optional): List of Aggregation
            Feature types to apply.

                Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "n_unique", "mode"]

        allowed_paths (list[list[str]]): Allowed table paths on which to make
            features.

        max_depth (int) : Maximum allowed depth of features.

        ignore_tables (list[str], optional): List of tables to
            blacklist when creating features.

        ignore_columns (dict[str -> list[str]], optional): List of specific
            columns within each table to blacklist when creating features.

        where_primitives (list[str or PrimitiveBase], optional):
            List of Primitives names (or types) to apply with where clauses.

                Default:

                    ["count"]

        max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

        features_only (bool, optional): If True, returns the list of
            features without calculating the feature matrix.

        save_progress (str, optional): Path to save intermediate computational results.

        return_column_types (list[Column] or str, optional): Types of
                columns to return. If None, default to
                Numeric, Discrete, and Boolean. If given as
                the string 'all', use all available column types.
        context [SparkSession]:the context of Spark 

    '''
    if not isinstance(tableset, TableSet):
        tableset = TableSet("dfs", tables, relationships)

    if context is None:
        context = SparkSession \
            .builder \
            .appName("dfs") \
            .enableHiveSupport()\
            .getOrCreate()

    dfs_object = DeepFeatureSynthesis(target_table, tableset,
                                      agg_primitives=agg_primitives,
                                      max_depth=max_depth,
                                      where_primitives=where_primitives,
                                      allowed_paths=allowed_paths,
                                      ignore_tables=ignore_tables,
                                      ignore_columns=ignore_columns,
                                      max_features=max_features,
                                      context = context)

    return dfs_object.build_features(verbose=verbose, return_column_types=return_column_types)

class DeepFeatureSynthesis(object):
    """Automatically produce features for a target table in an Tableset.

        Args:
            target_table_id (str): Id of table for which to build features.

            tableset (TableSet): Tableset for which to build features.

            agg_primitives (list[str or :class:`.primitives.`], optional):
                list of Aggregation Feature types to apply.

                Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]

            where_primitives (list[str or :class:`.primitives.PrimitiveBase`], optional):
                only add where clauses to these types of Primitives

                Default:

                    ["count"]

            max_depth (int, optional) : maximum allowed depth of features.
                Default: 2. If -1, no limit.

            max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

            allowed_paths (list[list[str]], optional): Allowed table paths to make
                features for. If None, use all paths.

            ignore_tables (list[str], optional): List of tables to
                blacklist when creating features. If None, use all tables.

            ignore_columns (dict[str -> list[str]], optional): List of specific
                columns within each table to blacklist when creating features.
                If None, use all columns.

            context [SparkSession]:the context of Spark 
        """

    def __init__(self,
                target_table_id,
                tableset,
                agg_primitives=None,
                where_primitives=None,
                max_depth=2,
                max_features=-1,
                allowed_paths=None,
                ignore_tables=None,
                ignore_columns=None,
                context=None):

        if target_table_id not in tableset.table_dict:
            ts_name = tableset.id or 'table set'
            msg = 'Provided target table %s does not exist in %s' % (target_table_id, ts_name)
            raise KeyError(msg)
        if context is None:
            context = SparkSession \
                .builder \
                .appName("dfs") \
                .enableHiveSupport()\
                .getOrCreate()
        self.context = context

        # need to change max_depth to None because DFs terminates when  <0
        if max_depth == -1:
            max_depth = None
        self.max_depth = max_depth

        self.max_features = max_features

        self.allowed_paths = allowed_paths
        if self.allowed_paths:
            self.allowed_paths = set()
            for path in allowed_paths:
                self.allowed_paths.add(tuple(path))

        if ignore_tables is None:
            self.ignore_tables = set()
        else:
            if not isinstance(ignore_tables, list):
                raise TypeError('ignore_tables must be a list')
            assert target_table_id not in ignore_tables,\
                "Can't ignore target_table!"
            self.ignore_tables = set(ignore_tables)

        self.ignore_columns = defaultdict(set)
        if ignore_columns is not None:
            for eid, vars in ignore_columns.items():
                self.ignore_columns[eid] = set(vars)
        self.target_table_id = target_table_id
        self.ts = tableset

        if agg_primitives is None:
            agg_primitives = ['avg','count','kurtosis','skewness','stddev','min','max','sum']
        self.agg_primitives = []
        agg_prim_dict = ['avg','count','kurtosis','skewness','stddev','min','max','sum']
        for a in agg_primitives:
            if isinstance(a,str):
                if a.lower() not in agg_prim_dict:
                    raise ValueError("Unknown aggregation primitive {}. ".format(a))
            self.agg_primitives.append(a.lower())

        if where_primitives is None:
            where_primitives = ['count']
        self.where_primitives = []
        for p in where_primitives:
            if isinstance(p,str):
                if p.lower() not in agg_prim_dict:
                    raise ValueError("Unknown aggregation primitive {}. ".format(p))
            self.where_primitives.append(p)
    
    def build_features(self, return_column_types=None, verbose=False):
        """Automatically builds feature definitions for target
            table using Deep Feature Synthesis algorithm

        Args:
            return_column_types (list[Column] or str, optional): Types of
                columns to return. If None, default to
                Numeric, Discrete, and Boolean. If given as
                the string 'all', use all available column types.

            verbose (bool, optional): If True, print progress.

        Returns:
            list[BaseFeature]: Returns a list of
                features for target table, sorted by feature depth
                (shallow first).
        """
        all_features = {}
        for e in self.ts.tables:
            if e not in self.ignore_tables:
                all_features[e.id] = {}

        self.where_clauses = defaultdict(set)

        self._run_dfs(self.ts[self.target_table_id], [],
                      all_features, max_depth=self.max_depth)

        return all_features

    
    def _run_dfs(self, table, table_path, all_features, max_depth):
        """
        create features for the provided table

        Args:
            table (Table): Table for which to create features.
            table_path (list[str]): List of table ids.
            all_features (dict[Table.id -> dict[str -> Column]]):
                Dict containing a dict for each table. Each nested dict
                has features as values with their ids as keys.
            max_depth (int) : Maximum allowed depth of features.
        """
        if max_depth is not None and max_depth < 0:
            return

        table_path.append(table.id)
        """
        Step 1 - Recursively build features for each table in a backward relationship
        """
        backward_tables = self.ts.get_backward_tables(table.id)
        backward_tables = [b_id for b_id in backward_tables if b_id not in self.ignore_tables]
        for b_table_id in backward_tables:
            # if in path, we've already built features
            if b_table_id in table_path:
                continue

            if self.allowed_paths and tuple(table_path + [b_table_id]) not in self.allowed_paths:
                continue
            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(table=self.ts[b_table_id],table_path=list(table_path),all_features=all_features,max_depth=new_max_depth)

        """
        Step 2 - Create agg_feat features for all deep backward relationships
        """

        print("rfeat",backward_tables,table_path,table.id)
        backward = [r for r in self.ts.get_backward_relationships(table.id)
                   if r.child_table.id in backward_tables and
                   r.child_table.id not in self.ignore_tables]
        for r in backward:
            if self.allowed_paths and tuple(table_path + [r.child_table.id]) not in self.allowed_paths:
                continue
            self._build_agg_features(r=r,
                                     all_features=all_features,
                                     max_depth=max_depth)

        """
        Step 3 - Add idtable features
        """
        self._add_all_features(all_features, table)

    def _build_agg_features(self, r, all_features, max_depth=0):
        if max_depth is not None and max_depth < 0:
            return

        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1

        input_types = "numeric"

        features = self._features_by_type(all_features=all_features,
                                            table=r.child_table,
                                            max_depth=new_max_depth,
                                            column_type=input_types)

        # remove features in relationship path
        relationship_path = self.ts.find_backward_path(r.parent_table.id, r.child_table.id)

        features = [f for f in features if not self._feature_in_relationship_path(relationship_path, f)]
        _local_data_stat_df = None

        group_all = list()  
        group_all.append(r.child_column.id) 
        for agg_prim in self.agg_primitives:

            features_prim={f.id:agg_prim for f in features}
            
            _tmp_stat_df = r.child_table.raw_data.groupby(group_all).agg(features_prim)
            print(_tmp_stat_df.toPandas())
            # join df one by one
            if _local_data_stat_df is None:
                _local_data_stat_df = _tmp_stat_df
            else:
                _local_data_stat_df = _local_data_stat_df.join(_tmp_stat_df, group_all, how='left_outer')

        for column in _local_data_stat_df.columns:
            if column in group_all:
                continue
            index=column.find('(')
            new_column = column[:index+1]+r.child_table.id+'.'+column[index+1:]
            _local_data_stat_df = _local_data_stat_df.withColumnRenamed(column,new_column)

            _c = ctypes.Numeric(new_column, r.parent_table)
            all_features[r.parent_table.id][new_column] = _c
            r.parent_table.columns += [_c]

        r.parent_table.raw_data = r.parent_table.raw_data.join(_local_data_stat_df,\
            r.parent_table.raw_data[r.parent_column.id]==_local_data_stat_df[r.child_column.id],how='left_outer')

        for col in group_all:
            r.parent_table.raw_data = r.parent_table.raw_data.drop(_local_data_stat_df[col])

    def _add_all_features(self, all_features, table):
        """add all columns from the given table into features

        Args:
            all_features (dict[Table.id -> dict[str -> Column]]):
                Dict containing a dict for each table. Each nested dict
                has features as values with their ids as keys.
            table (Table): Table to calculate features for.
        """
        columns = table.columns
        ignore_columns = self.ignore_columns[table.id]
        for v in columns:
            if v.id in ignore_columns:
                continue
            all_features[table.id][v.id]=v

    def _features_by_type(self, all_features, table, max_depth,
                          column_type=None):

        selected_features = []

        if max_depth is not None and max_depth < 0:
            return selected_features

        for feat in all_features[table.id]:
            f = all_features[table.id][feat]
            if f.dtype == column_type:
                selected_features.append(f)

        return selected_features

    def _feature_in_relationship_path(self, relationship_path, feature):

        for relationship in relationship_path:
            if relationship.child_table.id == feature.table.id and \
               relationship.child_column.id == feature.id:
                return True

            if relationship.parent_table.id == feature.table.id and \
               relationship.parent_column.id == feature.id:
                return True

        return False
        