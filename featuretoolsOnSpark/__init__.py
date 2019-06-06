from __future__ import print_function,absolute_import
from table import *
from dfs import *
from tableset import *
from column_types import *
from relationship import *

agg_prim_default = ['avg','count','kurtosis','skewness','stddev','min','max','sum']

def print_agg_prims():
    print(agg_prim_default)