import argparse
import feature_encoding
import logging
import openml
import os
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
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/feature_encoding')
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

    mi_target_num = sklearn.feature_selection.mutual_info_classif(X, y, False)
    mi_target_nom = sklearn.feature_selection.mutual_info_classif(X, y, True)

    record_count = 0
    for idx, record in importances.iterrows():
        n_non_integer_values = feature_encoding.num_non_integer_values(X, idx)
        if n_non_integer_values > 0:
            continue
        if record_count >= per_task_limit:
            break

        mi_feats_num = sklearn.feature_selection.mutual_info_classif(X, X[:, idx], False)
        mi_feats_nom = sklearn.feature_selection.mutual_info_classif(X, X[:, idx], True)

        current = {
            'task_id': task_id,
            'feature_idx': idx,
            'feature_name': record['name'],
            'mi_target_numeric': mi_target_num[idx],
            'mi_target_nominal': mi_target_nom[idx],
            'mi_target_diff': mi_target_num[idx] - mi_target_nom[idx],
            'mi_feats_sum_numeric': sum(mi_feats_num),
            'mi_feats_sum_nominal': sum(mi_feats_nom),
            'mi_feats_sum_diff': sum(mi_feats_num) - sum(mi_feats_nom),
            'mi_feats_max_numeric': max(mi_feats_num),
            'mi_feats_max_nominal': max(mi_feats_nom),
            'mi_feats_max_diff': max(mi_feats_num) - max(mi_feats_nom),
            'is_nominal': record['data_type'] == 'nominal'
        }
        results.append(current)
        record_count += 1
    return results


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    experiment_suffix = ''
    if args.task_limit is not None:
        study.tasks = study.tasks[:args.task_limit]
        experiment_suffix += '__tasks_%d' % args.task_limit

    results = []
    for task_id in study.tasks:
        try:
            results += process_task(task_id, args.random_state, args.per_task_limit)
        except openml.exceptions.OpenMLServerException:
            pass

    frame = pd.DataFrame(results)
    frame.to_csv(os.path.join(args.output_dir, 'mutual_information%s.csv' % experiment_suffix))
    y = frame['is_nominal'].values
    del frame['is_nominal']
    # remove non-predictive info
    del frame['task_id']
    del frame['feature_idx']
    del frame['feature_name']

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
