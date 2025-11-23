from datetime import datetime

import cloudpickle
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
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
    )
    df_prepared = df.filter(regex='^(utm|device|geo)_.+').copy()
    df_prepared['target_action'] = create_target_action(df)

    df_prepared.drop_duplicates(inplace=True)

    x = df_prepared.drop(columns='target_action')
    y = df_prepared.target_action

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, make_column_selector())
    ])

    pipe = Pipeline(steps=[
        ('transformer', transformer),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])

    pipe.fit(x, y)
    accuracy = cross_val_score(pipe, x, y, cv=10).mean()
    roc_auc = roc_auc_score(y, pipe.predict_proba(x)[:, 1])
    print(accuracy, roc_auc)

    with open('data/pipeline.pkl', 'wb') as file:
        cloudpickle.dump({
            'metadata': Metadata(
                name='Target action prediction',
                author='Roman Zhiltsov',
                version=1,
                date=datetime.now(),
                type=type(pipe.named_steps['classifier']).__name__,
                accuracy=accuracy,
                roc_auc=roc_auc),
            'pipeline': pipe
        }, file)


def create_target_action(df: pd.DataFrame) -> pd.Series:
    actions = {'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
               'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
               'sub_submit_success', 'sub_car_request_submit_click'}

    return df.event_action.apply(lambda x: int(x in actions))


if __name__ == '__main__':
    pipeline()
