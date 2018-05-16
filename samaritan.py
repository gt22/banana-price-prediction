# %% imports
from gc import collect

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn import *
from stop_words import get_stop_words
from typing import List, Dict
import samaritan_funcs as fc
from scipy.sparse import hstack
from importlib import reload
from time import sleep


# %%
rmsle_list: Dict[str, List[float]] = {
    'linreg': [0.],
    'ridge': [0.],
    'lasso': [0.]
}


def filter_word(d, w, c='item_description'):
    return d[d[c].map(lambda x: w in x)]


# %% read data
df: pd.DataFrame = pd.read_csv('data/cleaned_train.csv')
df_test: pd.DataFrame = pd.read_csv('data/cleaned_test.csv')

# %% pre-processing
tokenizer = RegexpTokenizer(r'\w+')
slash_tokenizer = RegexpTokenizer(r'[^/]+')
en_stop: List[str] = get_stop_words('en')
stemmer = PorterStemmer()


def clean_text(t: str, tk=tokenizer) -> str:
    tokens = tk.tokenize(t.lower())
    tokens = [stemmer.stem(token) for token in tokens
              if token not in en_stop or len(token) >= 2 and not token.isnumeric()]
    return ' '.join(tokens)


def preprocess(d: pd.DataFrame, clean=False) -> pd.DataFrame:
    d: pd.DataFrame = d.copy()
    d.fillna('__nan__', inplace=True)
    if clean:
        print("Cleaning 'name'")
        d['name'] = d['name'].apply(clean_text)
        print("Cleaning 'category_name'")
        d['category_name'] = d['category_name'].apply(lambda t: clean_text(t, slash_tokenizer))
        print('Cleaning description')
        d['item_description'] = d['item_description'].apply(clean_text)

    d['text'] = d['name'] + d['category_name'] + d['item_description']
    d['brand_cat'] = d['brand_name'] + d['category_name']
    return d


# %%
nullify_nan = False
preprocesser = pipeline.make_pipeline(preprocessing.FunctionTransformer(preprocess, validate=False),
                                      fc.MeanTransformer('brand_name', nullifiy_nan=nullify_nan),
                                      fc.MeanTransformer('category_name', nullifiy_nan=nullify_nan),
                                      fc.MeanTransformer('brand_cat', nullifiy_nan=nullify_nan),
                                      fc.MeanTransformer('brand_name', nullifiy_nan=nullify_nan, agg_fun='median'),
                                      fc.MeanTransformer('category_name', nullifiy_nan=nullify_nan, agg_fun='median'),
                                      fc.MeanTransformer('brand_cat', nullifiy_nan=nullify_nan, agg_fun='median'))
df = preprocesser.fit_transform(df, np.log1p(df['price']))
fc.alert("Preprocessing complete")
# %%
name_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=25000, token_pattern='\w+', min_df=1)
text_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=25000, token_pattern='\w+', min_df=1,
                                                          ngram_range=(1, 2))
vectorizer = pipeline.make_union(
    fc.on_field('name', name_vectorizer),
    fc.on_field('text', text_vectorizer),
    fc.on_field(['shipping', 'item_condition_id'],
                preprocessing.FunctionTransformer(fc.to_records, validate=False),
                feature_extraction.DictVectorizer()),
    n_jobs=1)
sample_frame: pd.DataFrame = df
sample_vect = vectorizer.fit_transform(sample_frame[['name', 'text', 'shipping', 'item_condition_id']])
mean_cols = ['brand_name_mean', 'category_name_mean', 'brand_cat_mean',
             'brand_name_median', 'category_name_median', 'brand_cat_median']
sample_vect = hstack([sample_frame[mean_cols], sample_vect], format='csr')
del sample_frame
collect()
fc.alert("Vectorization complete")
# %% train/test split
(train_vect, valid_vect,
 train_label, valid_label) = model_selection.train_test_split(sample_vect,
                                                              np.log1p(df['price']),
                                                              test_size=.3,
                                                              random_state=6741)
# %% fit linear regression
param_grid = {
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
}
linreg = linear_model.LinearRegression()

ridge = model_selection.GridSearchCV(linear_model.Ridge(alpha=1, random_state=6741), param_grid)
lasso = model_selection.GridSearchCV(linear_model.Lasso(alpha=1, random_state=6741), param_grid)
# %#%
linreg.fit(train_vect, train_label)
fc.alert("linreg")
collect()
# %#%
ridge.fit(train_vect, train_label)
fc.alert("ridge")
collect()
# %#%
lasso.fit(train_vect, train_label)
fc.alert("lasso")
collect()
# %%


def loss(model, name):
    lst = rmsle_list[name]
    lst.append(fc.rmsle(np.exp(valid_label), np.exp(np.clip(model.predict(valid_vect), 0, np.inf))))
    print(name + ": ")
    print("RMSLE: " + str(lst[-1]))
    print("Prev: " + str(lst[-2]))
    print("Diff: " + str(lst[-1] - lst[-2]))
    print()
    open('rmsle_%s.log' % name, 'a').write(str(lst[-1]))


loss(linreg, "linreg")
loss(ridge, "ridge")
loss(lasso, "lasso")
# fc.alert("Linear regression complete")
# %%
train_pred = pd.DataFrame({
    'linreg': linreg.predict(valid_vect),
    'ridge': ridge.predict(valid_vect),
    'lasso': lasso.predict(valid_vect)
})
train_pred['blend'] = train_pred[['linreg', 'ridge', 'lasso']].agg('mean', axis=1)
print("RMSLE: " + str(fc.rmsle(np.exp(valid_label), np.exp(np.clip(train_pred['blend'], 0, np.inf)))))

# %%
df_test = preprocesser.transform(df_test)
# %%
test_vect = vectorizer.transform(df_test[['name', 'text', 'shipping', 'item_condition_id']])
test_vect = hstack([df_test[mean_cols], test_vect], format='csr')
collect()
fc.alert("Test vectorization complete")
# %% create submission
submit = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'linreg': linreg.predict(test_vect),
    'ridge': ridge.predict(test_vect),
    'lasso': lasso.predict(test_vect)
})
# %% calculate and submit
# submit['price'] = np.exp(np.clip(submit['lasso'], 0, np.inf))
submit['price'] = np.exp(np.clip(submit[['linreg', 'ridge']].agg('mean', axis=1), 0, np.inf))
fc.submit(submit[['sample_id', 'price']])
sleep(1)
print(fc.get_kaggle_diff())
