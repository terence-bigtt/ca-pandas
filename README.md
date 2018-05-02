# ca-pandas

A library for correspondance analysis and an interface for pandas.

## Usage: 
1. Basic usage:

```python
class CA:
    def __init__(self, data, columns=[], rows=[], quanti_sup=[], quali_sup=[], quanti_sup_names=[]):
        """
        :param data: dataset, as a list of I lists of J elements
        :param rows: a list of I elements containing the name labels of the I individuals
        :param columns: a list of J elements containing the name of the J variables
        :param quanti_sup: suplementary quantitative variables: list of I lists of M supplementatry variables
        :param quali_sup: supplementary qualitative variables: list of I lists of N supplementary qualitative variables
        """
```

2. PandAs wrapper:
```python
class CAPandas(CA):
    def __init__(self, ddf, columns=slice(0, -1), row_names=None, quanti_sup=slice(0), quali_sup=slice(0)):
        """
        :param ddf: initial dataframe with data, name column, supplementary quantitative variables.
        :param columns: sice function for selecting the data columns
        :param row_names: slice function for selecting the name column or None if in the index
        :param quanti_sup: slice function for quantitive variables
        :param quali_sup: slice function for supplementary qualitative variables
        """
```
