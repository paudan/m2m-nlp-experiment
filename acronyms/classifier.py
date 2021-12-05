import os
import joblib
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from catboost.core import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

SEED = 42

def catboost_classifier(train_data, labels):
    clf = CatBoostClassifier(verbose=False)
    clf.fit(train_data, labels)
    return clf

def random_forest_classifier(train_data, labels):
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    clf.fit(train_data, labels)
    return clf

def voting_classifier(X_train, y_train):
    major = X_train[y_train==0]
    minor = X_train[y_train==1]
    kf = KFold(n_splits=10, random_state=SEED, shuffle=True)
    classifiers = []
    i = 1
    for ind, train_index in kf.split(major):
        print(f'Training classifier for subsample {i}')
        df_sub = np.vstack([major[train_index], minor])
        y_sub = [0] * len(train_index) + [1] * minor.shape[0]
        clf = catboost_classifier(df_sub, y_sub)
        classifiers.append((f'clf{i}', clf))
        i += 1
    clf = VotingClassifier(estimators=classifiers, voting='soft')
    clf.fit(X_train, y_train)
    return clf

def weighted_catboost_classifier(train_data, labels):
    classes=np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    clf = CatBoostClassifier(verbose=False, class_weights=class_weights)
    clf.fit(train_data, labels)
    return clf

def plain_catboost_classifier(train_data, y_train, weighted=False):
    X_train = train_data.copy()
    X_train.fillna(value='NA', inplace=True)
    X_train = X_train.astype('category')
    params = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}
    kwargs = {}
    if weighted is True:
        classes=np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        kwargs = {'class_weights': class_weights}
    cb = CatBoostClassifier(verbose=False, cat_features=train_data.columns.tolist(), **kwargs)
    cb.grid_search(params, X=X_train, y=y_train)
    return cb

def xgb_classifier(train_data, labels):
    clf = XGBClassifier(learning_rate=0.01, n_estimators=100)
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    search = GridSearchCV(clf, param_grid=params, cv=2, n_jobs=1)
    search.fit(train_data, labels)
    return search.best_estimator_


def preprocess_data(X_train, y_train):
    train_data = X_train.copy()
    labels_train = y_train.copy()
    label_encoder = LabelEncoder().fit(labels_train)
    labels = label_encoder.transform(labels_train)
    categorical_features_mask = (train_data.dtypes == bool) | (train_data.dtypes == object)
    categorical_features = train_data.columns[categorical_features_mask]

    def encode_column(train_data, col):
        train_data.loc[pd.isnull(train_data[col]), col] = 'N/A'
        encoder = LabelEncoder()
        train_data[col] = encoder.fit_transform(train_data[col])
        return encoder

    column_encoders = dict()
    ohe = None
    if len(categorical_features) > 0:
        column_encoders = {x: encode_column(train_data, x) for x in categorical_features}
        ohe = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features_mask)],
                                remainder='passthrough', sparse_threshold=0)
        train_data = ohe.fit_transform(train_data)
    return train_data, labels, column_encoders, ohe, label_encoder

def preprocess_test_data(X_test, y_test, column_encoders: Dict[str, LabelEncoder], ohe: ColumnTransformer, label_encoder: LabelEncoder):
    def apply_encoder(encoder, data):
        enc_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        return data.apply(lambda x: enc_dict.get(x, 999999))  # Encode "unseen" values
    test_data = X_test.copy()
    labels_test = label_encoder.transform(y_test)
    categorical_features = column_encoders.keys()
    if len(categorical_features) > 0:
        for x in categorical_features:
            test_data[x] = apply_encoder(column_encoders[x], test_data[x])
        test_data = ohe.transform(test_data)
    return test_data, labels_test


def test_classifier(clf, X_test, y_test, column_encoders: Dict[str, LabelEncoder], ohe: ColumnTransformer, label_encoder: LabelEncoder):
    test_data, labels_test = preprocess_test_data(X_test, y_test, column_encoders, ohe, label_encoder)
    results = clf.predict(test_data).astype(int)
    results = pd.DataFrame({'predicted': label_encoder.inverse_transform(results), 'original': labels_test})
    try:
        probs = pd.DataFrame(clf.predict_proba(test_data), columns=['Class_{}'.format(i) for i in set(labels_test)])
        results = pd.concat([results.reset_index(drop=True), probs.reset_index(drop=True)], axis='columns')
    except AttributeError:
        pass
    return results.reset_index(drop=True)


def train_classifier(X_train, y_train, output_dir: str='.', X_test=None, y_test=None, classifier=catboost_classifier):
    train_data, labels, column_encoders, ohe, label_encoder = preprocess_data(X_train, y_train)
    categorical_features = list(column_encoders.keys())
    clf = classifier(train_data, labels)
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        joblib.dump(clf, os.path.join(output_dir, 'model.joblib'))
        joblib.dump((categorical_features, label_encoder, column_encoders, ohe),
                    os.path.join(output_dir, 'preprocess.joblib'))
    results = None
    if X_test is not None:
        results = test_classifier(clf, X_test, y_test, column_encoders, ohe, label_encoder)
    return clf, results, ohe.get_feature_names()