import argparse
import openml
import os
import pandas as pd
import sklearn.ensemble
import sklearn.impute
import sklearn.pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--task_idx', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/feature_encoding')
    parser.add_argument('--n_trees', type=int, default=16)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()


def run(args):
    study = openml.study.get_study(args.study_id, 'tasks')
    task_id = study.tasks[args.task_idx]
    # TODO: make this lazy, see issue in issue tracker.
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='median'),
        sklearn.ensemble.RandomForestClassifier(n_estimators=args.n_trees,
                                                random_state=args.random_state)
    )

    pipeline.fit(X, y)
    importances = pipeline.steps[-1][-1].feature_importances_
    if len(importances) != X.shape[1]:
        raise ValueError()

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

    output_dir = os.path.join(args.output_dir, str(task_id))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'feature_importances.csv'))


if __name__ == '__main__':
    run(parse_args())
