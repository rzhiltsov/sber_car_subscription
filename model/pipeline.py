from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from web.response import Metadata


def pipeline():
    df = pd.merge(
        pd.read_csv('data/ga_sessions.csv', dtype={'client_id': str}),
        pd.read_csv('data/ga_hits.csv'),
        on='session_id'
    ).set_index('client_id')
    df = filter_columns(df)
    df = drop_fraud(df)
    df = fill_blanks(df)
    df = drop_duplicates(df)

    x = df.drop(columns='target_action')
    y = df.target_action

    transformer = ColumnTransformer(transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'), make_column_selector())
    ])

    preprocessor = Pipeline(steps=[
        ('transformer', transformer)
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])

    accuracy = cross_val_score(pipe, x, y, cv=10).mean()
    pipe.fit(x, y)
    roc_auc = roc_auc_score(y, pipe.predict_proba(x)[:, 1])
    print(accuracy, roc_auc)

    joblib.dump({
            'metadata': Metadata(
                name='Target action prediction',
                author='Roman Zhiltsov',
                version=1,
                date=datetime.now(),
                type=type(pipe.named_steps['classifier']).__name__,
                accuracy=accuracy,
                roc_auc=roc_auc),
            'pipeline': pipe
    }, 'data/pipeline.pkl')


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df.hit_number,
        df.filter(regex='^(utm|device|geo)_.+'),
        create_target_action(df)
    ], axis=1)


def create_target_action(df: pd.DataFrame) -> pd.Series:
    actions = {'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
               'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
               'sub_submit_success', 'sub_car_request_submit_click'}

    return df.event_action.apply(lambda x: int(x in actions)).rename('target_action')


def outliers_upper_boundary(s: pd.Series) -> float:
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    iqr = q75 - q25

    return q75 + 1.5 * iqr


def drop_fraud(df: pd.DataFrame) -> pd.DataFrame:
    boundary = outliers_upper_boundary(df.hit_number)

    return df.drop(index=df.hit_number.loc[lambda x: x > boundary].index).drop(columns='hit_number')


def fill_blanks(df: pd.DataFrame) -> pd.DataFrame:
    for column in df:
        mode = df[column].mode()[0]
        df[column] = df[column].fillna(mode)

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


if __name__ == '__main__':
    pipeline()
