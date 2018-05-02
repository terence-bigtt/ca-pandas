from scipy.stats import chi2
import numpy as np
from numpy.linalg import svd
from matplotlib import pyplot as plt
from functools import reduce

def flatten(seq): return reduce(lambda a, b: a + b, seq, [])

def tofloat(x): return float(x)

vtofloat = np.vectorize(tofloat)

class CA:
    def __init__(self, data, columns=[], rows=[], quanti_sup=[], quali_sup=[], quanti_sup_names=[]):
        """
        :param data: dataset, as a list of I lists of J elements
        :param rows: a list of I elements containing the name labels of the I individuals
        :param columns: a list of J elements containing the name of the J variables
        :param quanti_sup: suplementary quantitative variables: list of I lists of M supplementatry variables
        :param quali_sup: supplementary qualitative variables: list of I lists of N supplementary qualitative variables
        """
        self.data = np.matrix(data)
        self.columns = np.array(columns)
        self.rows = np.array(rows)
        self.quanti_sup = np.matrix(quanti_sup)
        self.quali_sup = np.matrix(quali_sup)
        self.quanti_sup_names = quanti_sup_names
        self._check_input()
        self._make()
        self._make_quali()
        self._make_quanti()
        self._chi2_test()

    def contribs(self, axis):
        lines, cols = self.contrib_lines(axis), self.contrib_columns(axis)
        return lines, cols

    def contrib_lines(self, axis):
        n_lines = self.data.shape[0]
        contribs = np.array(map(lambda i: self._contrib_to_line(i, axis), range(n_lines)))
        return contribs

    def contrib_columns(self, axis):
        n_col = self.data.shape[1]
        contribs = np.array(map(lambda j: self._contrib_to_col(j, axis), range(n_col)))
        return contribs

    def _contrib_to_line(self, i, axis):
        F = self.R
        Da = self.Da
        coords2 = F[i, axis] ** 2
        weight = self.r[i, 0]
        lambdai = Da[axis, axis]
        return coords2 * weight / lambdai ** 2

    def _contrib_to_col(self, j, axis):
        G = self.G
        Da = self.Da
        coords2 = G[j, axis] ** 2
        weight = self.c[j, 0]
        lambdai = Da[axis, axis]
        return coords2 * weight / lambdai ** 2

    def _chi2_test(self):
        degrees_of_freedom = (self.data.shape - np.array((1, 1))).prod()
        pf = lambda x: 1. - chi2.cdf(x, degrees_of_freedom)
        self.chi2Val = self.inertia * (self.data.sum())
        self.p = pf(self.chi2Val)

    def d_lines_chi2(self, i, l):
        return self.d_chi2_lines(self.P[i, :], self.P[l, :])

    def d_columns_chi2(self, i, l):
        return self.d_chi2_cols(self.P[:, i], self.P[:, l])

    def d_chi2_lines(self, pi, pj):
        xi = pi / pi.sum()
        xl = pj / pj.sum()
        delta = xi - xl
        g = self.Dr_sqrt * self.Dr_sqrt
        return (delta.T * g * delta).A1[0]

    def d_chi2_cols(self, pi, pj):
        xi = pi / pi.sum()
        xl = pj / pj.sum()
        delta = xi - xl
        g = self.Dr_sqrt * self.Dr_sqrt
        return (delta.T * g * delta).A1[0]

    def _make(self):
        P = self.data / self.data.sum()
        r = P.sum(axis=1)
        c = P.sum(axis=0).T
        self.r = r
        self.c = c
        Dr_sqrt = np.diag(1. / np.sqrt(r.A1))
        Dc_sqrt = np.diag(1. / np.sqrt(c.A1))
        self.Dr_sqrt = Dr_sqrt
        self.Dc_sqrt = Dc_sqrt
        S = Dr_sqrt * (P - r * (c.T)) * Dc_sqrt
        self.S = S
        self.P = P

        self.U, self.Da, self.V = svd(S, full_matrices=False)
        self.Da = np.asmatrix(np.diag(self.Da))
        self.V = self.V.T

        # Standard coordinates of rows
        self.X = Dr_sqrt * self.U
        # Standard coordinates of columns
        self.Y = Dc_sqrt * self.V
        # Principal coordinates of rows:
        self.F = self.X * self.Da
        # Principal coordinates of columns:
        self.G = self.Y * self.Da
        self.inertia = sum([(P[i, j] - r[i, 0] * c[j, 0]) ** 2 / (r[i, 0] * c[j, 0])
                            for i in range(self.data.shape[0])
                            for j in range(self.data.shape[1])])
        self.eigenvals = np.diag(self.Da) ** 2
        self.inertia_per = np.diag(self.Da) ** 2 / self.inertia
        self.inertia_cum = self.inertia_per.cumsum()
        # conventions from http://www.mathematica-journal.com/data/uploads/2010/09/Yelland.pdf
        self.R = self.F
        self.C = self.Y

    def _make_quali(self):
        Fquali = {}
        Xquali = {}
        for i in range(0, self.quali_sup.shape[1]):
            quali = self.quali_sup[:, i].A1
            vclas = list(set(quali))
            positions = {c: [j for j, v in enumerate(quali) if c == v] for c in vclas}
            newF = {}
            newX = {}
            for category in positions.keys():
                position = positions.get(category)
                meanF = sum([self.F[j] for j in position]) / float(len(position))
                meanX = sum([self.X[j] for j in position]) / float(len(position))
                newF.update({category: meanF})
                newX.update({category: meanX})
            Fquali.update({i: newF})
            Xquali.update({i: newX})
        self.Fquali = Fquali
        self.Xquali = Xquali
        names = []
        quali_axes = self.Fquali.keys()
        indexes = flatten([[(q, v) for v in self.Fquali.get(q)] for q in quali_axes])
        self.FqualiData = np.matrix(map(lambda ind: self.Fquali.get(ind[0]).get(ind[1]).A1, indexes))
        self.FqualiNames = indexes

    def _make_quanti(self):
        Da = self.Da
        cDa = np.diag((1. / Da.diagonal()).A1)
        Psupp = self.quanti_sup
        if Psupp.size != 0:
            dr = np.diag((1. / np.sqrt(Psupp.sum(axis=1))).A1)
            dc = np.diag((1. / np.sqrt(Psupp.sum(axis=0))).A1)
            print('shape dr: ' + str(dr.shape))
            print('shape dc: ' + str(dc.shape))
            print('shape Psup: ' + str(Psupp.shape))
            print('shape X: ' + str(self.X.shape))
            self.Gquanti = (self.X.T * Psupp * dc * dc).T
            self.Yquanti = self.Gquanti * cDa
        else:
            self.Yquanti = Psupp
            self.Gquanti = Psupp

    def _check_input(self):
        data_shape = self.data.shape
        quali_shape = self.quali_sup.shape
        quanti_shape = self.quanti_sup.shape

        if len(self.columns) == 0:
            self.columns = np.array(['v' + str(i) for i in range(len(self.data.T))])
        if len(self.rows) == 0:
            self.rows = np.array(['i' + str(i) for i in range(len(self.data))])
        if data_shape[0] != self.rows.shape[0]:
            raise ValueError("Incompatible rows and data")
        if data_shape[1] != self.columns.shape[0]:
            raise ValueError("Incompatible columns and data")
        if quanti_shape != (1, 0) and quanti_shape[0] != data_shape[0]:
            print(quanti_shape, data_shape)
            print("Incompatible quanti_sup and data, omitting quanti_sup")
            self.quanti_sup = np.matrix([])
        if quali_shape != (1, 0) and quali_shape[0] != data_shape[0]:
            print(quali_shape, data_shape)
            print("Incompatible quali_sup and data, omitting quali_sup")
            self.quali_sup = np.matrix([])
        self.data = vtofloat(self.data)
        if np.prod(self.quanti_sup.shape) != 0:
            print(np.prod(self.quanti_sup.shape) == 0, np.prod(self.quanti_sup.shape))
            self.quanti_sup = vtofloat(self.quanti_sup)

    def _plot_axis(self, X, axis, labels, color, marker):
        axis0 = axis[0]
        axis1 = axis[1]
        x, y = X[:, axis0].A1, X[:, axis1].A1
        plt.scatter(x, y, c=color, marker=marker, linewidths=0)
        xy = list(zip(x, y))
        if labels is not None:
            for i, n in enumerate(labels):
                plt.annotate(n, xy[i], color=color)

    def _plot_data(self, X, Y, axis=[0, 1], labels=True, max_label=100, color=['k', 'r'], markers=['.', '.'],
                   show=['all']):
        axis0 = axis[0]
        axis1 = axis[1]
        inertias = round(self.inertia_per[axis0] * 100, 2), round(self.inertia_per[axis1] * 100, 2)

        if 'all' in show or 'row' in show:
            self._plot_axis(X, axis, self.rows[:max_label], color[0], markers[0])

        if 'all' in show or 'column' in show:
            self._plot_axis(Y, axis, self.columns[:max_label], color[1], markers[1])

        plt.xlabel('Axis {} - {} %'.format(str(axis0), inertias[0]))
        plt.ylabel('Axis {} - {} %'.format(str(axis1), inertias[1]))

    def plot(self, axis=[0, 1], labels=True, max_label=100, color=['k', 'r'], markers=['.', '.'], show=['all']):
        self._plot_data(self.F, self.G, axis, labels, max_label, color, markers, show)

    def plot_quanti(self, axis=[0, 1], color='b', marker='o'):
        self._plot_axis(self.Gquanti, axis, self.quanti_sup_names, color, marker)

    def plot_quali(self, axis=[0, 1], color='b', marker='o'):
        self._plot_axis(self.FqualiData, axis, map(lambda n: n[1], self.FqualiNames), color, marker)

    def plot_barycenter_line(self, axis=[0, 1], labels=True, max_label=100, color=['k', 'r'], markers=['.', '.'],
                             show='all'):
        self._plot_data(self.F, self.Y, axis, labels, max_label, color, markers, show)

    def plot_barycenter_column(self, axis=[0, 1], labels=True, max_label=100, color=['k', 'r'], markers=['.', '.'],
                               show='all'):
        self._plot_data(self.X, self.G, axis, labels, max_label, color, markers, show)


class CAPandas(CA):
    def __init__(self, ddf, columns=slice(0, -1), row_names=None, quanti_sup=slice(0), quali_sup=slice(0)):
        """
        :param ddf: initial dataframe with data, name column, supplementary quantitative variables.
        :param columns: sice function for selecting the data columns
        :param row_names: slice function for selecting the name column or None if in the index
        :param quanti_sup: slice function for quantitive variables
        :param quali_sup: slice function for supplementary qualitative variables
        """
        df = ddf.replace({0: None}).dropna()
        data = df.T[columns].T.values
        data_cols = list(df.columns[columns])
        data_rows = list(df.index)
        if row_names is not None:
            data_rows = df[row_names]

        quanti_s = df.T[quanti_sup].T.values
        quali_s = df.T[quali_sup].T.values
        self.quali_s = quali_s
        self.quanti_s = quanti_s
        quanti_s_names = df.columns[quanti_sup]
        CA.__init__(self, data, data_cols, data_rows, quanti_s, quali_s, quanti_s_names)
