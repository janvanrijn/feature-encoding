import openml
import pandas as pd
import sklearn.ensemble
import sklearn.impute
import sklearn.pipeline


def feature_importance_on_openml_task(task: openml.tasks.OpenMLSupervisedTask,
                                      n_trees: int,
                                      random_state: int) -> pd.DataFrame:
    X, y = task.get_X_and_y()

    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='median'),
        sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees,
                                                random_state=random_state)
    )

    pipeline.fit(X, y)
    importances = pipeline.steps[-1][-1].feature_importances_
    if len(importances) != X.shape[1]:
        raise ValueError('Did not obtain feature importance for all attributes,'
                         'probably due to constant missing val')

    features = task.get_dataset().features

    results = list()
    for idx, importance in enumerate(importances):
        results.append({
            'idx': idx,
            'importance': importance,
            'name': features[idx].name,
            'data_type': features[idx].data_type
        })
    df = pd.DataFrame(results)
    df = df.set_index('idx')
    return df
