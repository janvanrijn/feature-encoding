import argparse
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
    parser.add_argument('--task_limit', type=int, default=None)
    return parser.parse_args()


def process_task(task_id):
    logging.info('Starting on task %d' % task_id)
    results = []
    task = openml.tasks.get_task(task_id)

    X, y = task.get_X_and_y()
    imp = sklearn.impute.SimpleImputer(strategy='constant', fill_value=-999999)
    X = imp.fit_transform(X)

    res_num = sklearn.feature_selection.mutual_info_classif(X, y, False)
    res_nom = sklearn.feature_selection.mutual_info_classif(X, y, True)
    nominals = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    for i in range(X.shape[1]):
        current = {
            'mutual_info_numeric': res_num[i],
            'mutual_info_nominal': res_nom[i],
            'mutual_info_diff': res_num[i] - res_nom[i],
            'is_nominal': i in nominals
        }
        results.append(current)
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
            results += process_task(task_id)
        except openml.exceptions.OpenMLServerException:
            pass

    frame = pd.DataFrame(results)
    y = frame['is_nominal'].values
    del frame['is_nominal']

    classifiers = {
        'dummy': sklearn.dummy.DummyClassifier(),
        'dt': sklearn.tree.DecisionTreeClassifier(random_state=0),
        'rf': sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=100)
    }

    for name, clf in classifiers.items():
        scores = sklearn.model_selection.cross_val_score(clf, frame.values, y, cv=5)
        total = sum(scores) / len(scores)
        print(name, total)


if __name__ == '__main__':
    run(parse_args())
