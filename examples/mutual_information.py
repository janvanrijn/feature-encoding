import argparse
import feature_encoding
import logging
import openml
import pandas as pd
import sklearn.dummy
import sklearn.ensemble
import sklearn.impute
import sklearn.feature_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--per_task_limit', type=int, default=10)
    parser.add_argument('--task_limit', type=int, default=None)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()


def process_task(task_id, random_state, per_task_limit):
    logging.info('Starting on task %d' % task_id)
    n_trees = 16
    results = []
    task = openml.tasks.get_task(task_id)
    importances = feature_encoding.feature_importance_on_openml_task(task, n_trees, random_state)
    importances.sort_values(by=['importance'], ascending=False, inplace=True)

    X, y = task.get_X_and_y()
    imp = sklearn.impute.SimpleImputer(strategy='constant', fill_value=-999999)
    X = imp.fit_transform(X)

    res_num = sklearn.feature_selection.mutual_info_classif(X, y, False)
    res_nom = sklearn.feature_selection.mutual_info_classif(X, y, True)

    record_count = 0
    for idx, record in importances.iterrows():
        if record_count >= per_task_limit:
            break
        current = {
            'mutual_info_numeric': res_num[idx],
            'mutual_info_nominal': res_nom[idx],
            'mutual_info_diff': res_num[idx] - res_nom[idx],
            'is_nominal': record['data_type'] == 'nominal'
        }
        results.append(current)
        record_count += 1
    return results


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    if args.task_limit is not None:
        study.tasks = study.tasks[:args.task_limit]

    results = []
    for task_id in study.tasks:
        try:
            results += process_task(task_id, args.random_state, args.per_task_limit)
        except openml.exceptions.OpenMLServerException:
            pass
        except ValueError:
            pass

    frame = pd.DataFrame(results)
    y = frame['is_nominal'].values
    del frame['is_nominal']

    classifiers = {
        'dummy': sklearn.dummy.DummyClassifier(),
        'dt': sklearn.tree.DecisionTreeClassifier(random_state=args.random_state),
        'rf': sklearn.ensemble.RandomForestClassifier(random_state=args.random_state, n_estimators=100)
    }

    for name, clf in classifiers.items():
        scores = sklearn.model_selection.cross_val_score(clf, frame.values, y, cv=5)
        total = sum(scores) / len(scores)
        print(name, total)


if __name__ == '__main__':
    run(parse_args())
