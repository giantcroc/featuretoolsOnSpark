from __future__ import print_function,absolute_import

from featuretoolsOnSpark.tableset import *
from featuretoolsOnSpark.table import *
from featuretoolsOnSpark.column_types import *
from featuretoolsOnSpark.relationship import *
from featuretoolsOnSpark.dfs import *
from featuretoolsOnSpark.version import __version__
from featuretoolsOnSpark.util import *

agg_prim_default = ['avg','count','kurtosis','skewness','stddev','min','max','sum']

def print_agg_prims():
    print(agg_prim_default)