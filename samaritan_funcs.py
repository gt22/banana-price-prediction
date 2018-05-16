from math import sqrt
from operator import itemgetter as get
from os import system, remove
from subprocess import run, PIPE
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from websocket import create_connection
from pandas import DataFrame, Series, merge
from typing import Tuple, List, Union, Callable, Iterable
from sklearn.base import BaseEstimator, TransformerMixin


def rmsle(y_true, y_pred) -> float:
    return sqrt(mean_squared_log_error(y_true, y_pred))


def init_websocket():
    ws = create_connection("ws://10.8.0.1:8080/alert")
    return ws


def alert(msg: str='Computation complete', ws: bool=True, usr: str='Admin') -> str:
    if not ws:
        system('espeak "%s"' % msg)
        return "Alerted"
    else:
        uadab = init_websocket()
        uadab.send("alert$%s:%s" % (usr, msg.replace(' ', ':')))
        rep = uadab.recv()
        uadab.shutdown()
        return rep


def on_field(f: Union[str, List[str]], *vec):
    return make_pipeline(FunctionTransformer(get(f), validate=False), *vec)


def to_records(df: DataFrame):
    return df.to_dict(orient='records')


def get_kaggle_score(best_fun=min) -> Tuple[float, float]:
    response: str = run(["kaggle", "competitions", "submissions", "-c", "banana-price-prediction"], stdout=PIPE) \
        .stdout.decode('utf-8')
    lines: list = response.split('\n')[:-1]
    cols: list = lines[0]
    pub_score = cols.index('publicScore')  # date field contains space, so +1
    scores = [line[pub_score:line.index(' ', pub_score)] for line in lines[2:]]
    scores = [float(s) for s in scores if s != 'None']
    cur_score = scores[0]
    best_score = best_fun(scores[1:])
    return cur_score, best_score


def get_kaggle_diff(best_fun=min) -> float:
    cur, best = get_kaggle_score(best_fun)
    return cur-best


def submit(df: Union[DataFrame, Series], filename: str = 'data/submit.csv', index=False):
    df.to_csv(filename, index=index, header=True)
    system('gzip -f "%s"' % filename)
    system("kaggle competitions submit -c banana-price-prediction -f '%s.gz' -m ''" % filename)
    remove(filename)


class MeanTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols: Union[str, List[str]]=None, target=None, nullifiy_nan: bool=False,
                 agg_fun: Union[str, Callable[[Iterable[float]], float]]='mean'):
        if isinstance(cols, str):
            cols = [cols]
        if len(cols) > 1 and nullifiy_nan:
            raise ValueError("nullify_nan can only be used with one col")
        self.cols = cols
        self.target = target
        self.nullify_nan = nullifiy_nan
        self.agg_fun = agg_fun
        self.mean = None
        self.name = '_'.join(cols + [agg_fun])

    def fit(self, x: DataFrame, y=None):
        if y is None:
            if self.target is None:
                raise ValueError("Either y or target should be set")
            else:
                y = self.target
        cols = self.cols
        data: DataFrame = x.copy()
        data.fillna('__nan__', inplace=True)
        data['__target__'] = Series(y)
        mean = data[cols + ['__target__']].groupby(cols).agg(self.agg_fun)
        if self.nullify_nan:
            mean.loc['__nan__'] = 0
        mean = mean.reset_index().rename(columns={'__target__': self.name})
        self.mean = mean
        return self

    def transform(self, x):
        data = x.copy()
        if self.name in data.columns:
            data.drop(self.name, axis=1, inplace=True)
        data = merge(data, self.mean, on=self.cols, how='left')
        data.loc[:, self.name].fillna(0, inplace=True)
        return data
