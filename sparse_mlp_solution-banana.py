# In[1]:
import os
from contextlib import contextmanager
from operator import itemgetter
import time
import gc

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')
# In[6]:


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.2f} seconds'.format(name, time.time() - t0))


def preprocess(df):
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    df['category_name'].fillna('unk', inplace=True)
    return df[['name', 'text', 'shipping', 'item_condition_id', 'category_name']]


def on_field(f, *vec):
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def to_records(df):
    return df.to_dict(orient='records')


def tfidf_fabric(x=1): 
    return Tfidf(max_features=15000, token_pattern='\w+', ngram_range=(1, x))


def fit_predict(xs, y_train, X_final_test):
    x_train, x_test = xs

    with tf.Session(graph=tf.Graph()) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(x_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(256, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=5e-3))
        for i in range(1):
            with timer('epoch {0}'.format(i + 1)):
                model.fit(x=x_train, y=y_train, batch_size=2**11, epochs=1, verbose=0)
        val = model.predict(x_test)[:, 0]
        tst = model.predict(X_final_test)[:, 0]
        return val, tst


# In[15]:


PREFIX = 'data/'
train = pd.read_csv(PREFIX + 'train_items.csv')
train = train[train['price'] > 0].reset_index(drop=True)

test = pd.read_csv(PREFIX + 'test_items.csv')
sid = test.sample_id.values
del test['sample_id']


# In[16]:


# noinspection PyTypeChecker
vectorizer = make_union(
        on_field('name', tfidf_fabric()),
        on_field('text', tfidf_fabric()),
        on_field(['shipping', 'item_condition_id', 'category_name'], 
                 FunctionTransformer(to_records, validate=False), DictVectorizer()))


# In[17]:


cv = KFold(n_splits=10, shuffle=True, random_state=42)
train_ids, valid_ids = next(cv.split(train))
train, valid = train.iloc[train_ids], train.iloc[valid_ids]


# In[18]:


y_train = np.log1p(train['price'].values.reshape(-1, 1))
y_valid = valid['price'].values

X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
X_final = vectorizer.transform(preprocess(test)).astype(np.float32)

print('X_train: {0} of {1}'.format(X_train.shape, X_train.dtype))
print('y_train: {0} of {1}'.format(y_train.shape, y_train.dtype))

print('X_valid: {0} of {1}'.format(X_valid.shape, X_valid.dtype))
print('y_valid: {0} of {1}'.format(y_valid.shape, y_valid.dtype))

print('X_final: {0} of {1}'.format(X_final.shape, X_final.dtype))

del train
del valid
del test

gc.collect()


# In[22]:


assert np.isnan(X_train.data).sum() == np.isinf(X_train.data).sum() == 0
assert np.isnan(X_valid.data).sum() == np.isinf(X_valid.data).sum() == 0
assert np.isnan(X_final.data).sum() == np.isinf(X_final.data).sum() == 0


# In[23]:


Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
Xb_final = X_final.astype(bool).astype(np.float32)

xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
xt = [Xb_final, X_final] * 2
print(len(xs), len(xt))


# In[24]:


preds_val = []
preds_tst = []
for i in range(len(xs)):
    x_trn = xs[i][0]
    x_tst = xs[i][1]
    x_fnl = xt[i]
    y_pred_val, y_pred_tst = fit_predict([x_trn, x_tst], y_train, x_fnl)
    preds_val.append(y_pred_val)
    preds_tst.append(y_pred_tst)
    y_pred_val = np.expm1(y_pred_val.reshape(-1, 1))[:, 0]
    print('Valid RMSLE: {:.4f} \n'.format(np.sqrt(mean_squared_log_error(y_valid, y_pred_val))))


# In[25]:


preds_val_ = np.mean(preds_val, axis=0)
preds_tst_ = np.mean(preds_tst, axis=0)


# In[26]:


y_pred = np.expm1(preds_val_.reshape(-1, 1))[:, 0]
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(y_valid, y_pred))))


# In[27]:


pd.DataFrame({'sample_id':sid, 
              'price': np.expm1(preds_tst_.reshape(-1, 1))[:, 0]}).to_csv('sparse_mlp.csv', index=False)

