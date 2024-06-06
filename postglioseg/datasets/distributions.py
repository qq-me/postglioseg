"""Whatever"""
import random
from typing import Optional, Sequence
import numpy as np
import polars as pl

def random_normal_in_range(low, high, size):
    """normal distribution rougly l to h"""
    high-=low
    return low + (np.random.normal(4, 1, size))/(8/high)

def one_hot(label:str, label_int:dict[str, int]):
    arr = list(range((len(label_int))))
    arr[label_int[label]] = 1
    return arr

def one_hot_if_three_plus(label:str, label_int:dict[str, int]):
    if len(label_int) < 3: return label_int[label]
    return one_hot(label, label_int)

class DistributionLabeled:
    def __init__(self, features:Optional[list] = None, n_features:Optional[int] = None, class_col:str = 'class', label_enc = one_hot_if_three_plus):
        if n_features is None and features is None: raise ValueError("Either `features` or `n_features` must be specified.")

        if n_features is None: n_features = len(features) # type:ignore
        if features is None: features = [f'x{i}' for i in range(n_features)]

        self.n_features = n_features
        """How many features in each sample"""

        self.features: list[str] = features
        """list of feature names, e.g. `['age', 'income', 'height']`. If `n_features` was specified, it is `['x0', 'x1', ...]`"""

        self.class_col: str = class_col
        """name of the class column"""

        self.classes: list[str] = []
        """List of all unique classes that gets populated as samples are added."""

        self.feature_int = {v:i for i,v in enumerate(self.features)}
        """mapping from feature name to integer index, e.g. `{'x0':0, 'x1':1, 'x2':2}`"""

        self.int_feature = {i:v for i,v in enumerate(self.features)}
        """mapping from integer index to feature name, e.g. `{0:'x0', 1:'x1', 2:'x2'}`"""

        self.df = pl.DataFrame(schema = [self.class_col] + self.features)

        self.label_enc = label_enc

    def copy(self):
        ds = DistributionLabeled(self.features, None, self.class_col)
        ds.df = self.df
        ds.classes = self.classes
        return ds

    def add_class(self, cls):
        if cls not in self.classes: self.classes.append(cls)

    def add_point(self, cls, features:Sequence[float]):
        """Add a single point with class `cls` and features `features`."""
        self.add_class(cls)
        new_rows = {self.class_col: cls, **{self.features[i]: x_num for i, x_num in enumerate(features)}}
        new_df = pl.DataFrame(new_rows)
        self.df = pl.concat([self.df, new_df], how='vertical_relaxed')

    def add_arr(self, cls, arr):
        """adds array in a shape of (num, self.n_features)"""
        self.add_class(cls)
        new_rows = {self.class_col: cls, **{self.features[i]: arr[:,i] for i in range(self.n_features)}}
        new_df = pl.DataFrame(new_rows)
        self.df = pl.concat([self.df, new_df], how='vertical_relaxed')

    def add_func(self, num, cls, func, **kwargs):
        """Add `num` samples with class `cls` with the given function.

        `func` tales in two arguments: `np.zeros((num, self.n_features))` and `**kwargs`"""
        samples = func(np.zeros((num, self.n_features)), **kwargs)
        self.add_arr(cls, samples)

    def add_line(self, num, cls, start:Sequence[float], end:Sequence[float]):
        """Create a line of `num` samples with start at `start` and end at `end`."""
        samples = np.zeros((num, self.n_features))
        for i in range(self.n_features):
            samples[:,i] = np.linspace(start[i], end[i], num)
        self.add_arr(cls, samples)

    def add_dist(self, num:int, cls, ranges:Sequence[Sequence[float]], dist_func):
        """Add `num` samples with class `cls` with the given distribution function. `ranges` should contain the ranges for each dimension.

        `dist_func` should take in 3 arguments: lower bound, upper bound, and shape / number of samples"""
        samples = np.zeros((num, self.n_features))
        for i in range(self.n_features):
            samples[:,i] = dist_func(ranges[i][0], ranges[i][1], num)

        self.add_arr(cls, samples)


    def add_uniform(self, num:int, cls, ranges:Sequence[Sequence[float]]):
        """Add `num` uniformly sampled elements with class `cls`. `ranges` should contain the ranges for each dimension."""
        self.add_dist(num, cls, ranges, np.random.uniform)

    def add_triangular(self, num:int, cls, ranges:Sequence[Sequence[float]], mode = 0.5):
        """Add `num` triangularly sampled elements with class `cls`, with mode being the `mode` of the range. `ranges` should contain the ranges for each dimension."""
        self.add_dist(num, cls, ranges, lambda low, high, num: np.random.triangular(low, low+((high-low)*mode), high, num))

    def add_normal(self, num:int, cls, ranges:Sequence[Sequence[float]]):
        """Add `num` normally sampled elements with class `cls`. `ranges` should contain the ranges for each dimension."""
        self.add_dist(num, cls, ranges, random_normal_in_range)

    @property
    def class_int(self):
        """Mapping from class string to integer encoding, e.g. `{'cat':0, 'bird':1, 'dog':2}`"""
        return {cls: i for i, cls in enumerate(self.classes)}

    @property
    def int_class(self):
        """Mapping from integer encoding to class string, e.g. `{0:'cat', 1:'bird', 2:'dog'}`"""
        return {i: cls for i, cls in enumerate(self.classes)}

    def get_classes_col(self) -> pl.Series:
        """Returns the `classes` column encoded into integers"""
        return self.df[self.class_col]

    def get_classes(self) -> list[int]:
        """Returns the `classes` column encoded into integers"""
        return self.get_classes_col().to_list()

    def get_classes_as_ints(self) -> list[int]:
        """Returns the `classes` column encoded into integers"""
        return [self.class_int[cls] for cls in self.df[self.class_col]]

    def get_feature_col(self, feature:str) -> pl.Series:
        """Returns the feature column."""
        return self.df[feature]

    def get_feature(self, feature:str):
        return self.get_feature_col(feature).to_list()

    def get_feature_col_by_index(self, feature_ind:int):
        """Returns the feature column by feature index."""
        return self.df[self.features[feature_ind]]

    def get_feature_col_index(self, feature:str):
        """Returns the feature column."""
        return self.df.get_column_index(feature)

    def plt_scatter(self, show = False, fig = False, features: list[str] = None, **kwargs):
        if features is None: features = self.features[:2]
        if len(features) != 2: raise ValueError("Must specify two features to plot.")

        import matplotlib.pyplot as plt
        if fig:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(x = self.get_feature_col(features[0]), y = self.get_feature_col(features[1]), c = self.get_classes_as_ints(), **kwargs)
            if show: plt.show()
            return fig
        else:
            plt.scatter(x = self.get_feature_col(features[0]), y = self.get_feature_col(features[1]), c = self.get_classes_as_ints(), **kwargs)#type:ignore
            if show: plt.show()

    def alt_scatter(self, features: list[str] = None):
        if features is None: features = self.features[:2]
        if len(features) != 2: raise ValueError("Must specify two features to plot.")

        import altair as alt
        return alt.Chart(self.df).mark_point().encode(x=features[0], y=features[1], color=self.class_col).interactive()#type:ignore

    def px_scatter(self, features: list[str] = None, **kwargs):
        if features is None: features = self.features[:2]
        if len(features) != 2: raise ValueError("Must specify two features to plot.")

        import plotly.express as px
        return px.scatter(self.df, x=self.get_feature_col_index(features[0]), y=self.get_feature_col_index(features[1]), color=0, **kwargs)

    def scatter(self, features: list[str] = None, encode_classes = True):
        """Returns [cols], classes"""
        if features is None: features = self.features[:2]
        return [list(self.get_feature(f)) for f in features], (self.get_classes_as_ints() if encode_classes else self.get_classes())

    def __getitem__(self, i:int | list[int] | slice):
        arr = self.df[i].to_numpy()
        # numpy array with obj dtype with class being the first column
        # for example [['cat', 4, 9.1], ['dog', 3, 2.5], ...]
        classes = np.array([self.label_enc(sample[0], self.class_int) for sample in arr], dtype=float)
        data = np.array([sample[1:] for sample in arr], dtype=float)
        return data, classes

    def one_hot(self,label:str):
        arr = list(range((len(self.classes))))
        arr[self.class_int[label]] = 1
        return arr

    def dataloader(self, batch_size, shuffle = True, remainder = True):
        """DataLoader-like generator, yielding batches of (labels, data)"""
        self.indexes = np.arange(len(self.df))
        self.cur = 0
        if shuffle: np.random.shuffle(self.indexes)
        for i in range(0, len(self.indexes), batch_size):
            self.cur += batch_size
            yield self[self.indexes[i:i+batch_size]]
        if remainder and self.cur < len(self.df):
            yield self[self.indexes[self.cur:]]

    def __len__(self):
        return len(self.df)

    def split(self, splits:list[float], shuffle = True):
        """Split the dataset into `splits` parts."""
        if sum(splits) != 1: raise ValueError(f"{splits} sum up to {sum(splits)}, they should sum up to 1.")
        dsets = [self.copy() for _ in range(len(splits))]
        indexes = list(range(len(self)))
        if shuffle: random.shuffle(indexes)

        index_splits = [int(splits[i] * len(indexes)) for i in range(len(splits))]
        cur = 0
        for i in range(len(splits)):
            dsets[i].df = self.df[indexes[cur : cur + min(index_splits[i], len(self.df))]]
            cur += index_splits[i]

        return dsets

if __name__ == "__main__":
    test_ds = DistributionLabeled(['x','y'])
    test_ds.add_uniform(100, 'cls1', [[0,1],[0,1]])
    test_ds.add_uniform(100, 'cls2', [[1,2],[1,2]])
    train, test = test_ds.split([0.9, 0.1])
    test.get_feature_col('x').to_list()
