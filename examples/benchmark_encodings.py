import argparse
import collections
import logging
import numpy as np
import openml
import os
import pandas as pd
import sklearnbot


def parse_args():
    parser = argparse.ArgumentParser()
    all_classifiers = sklearnbot.config_spaces.get_available_config_spaces(True)
    parser.add_argument('--classifier_name', type=str, choices=all_classifiers, default='decision_tree',
                        help='the classifier to run')
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/feature_encoding')
    parser.add_argument('--feature_cutoff', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    configuration_space = sklearnbot.config_spaces.get_config_space(args.classifier_name,
                                                                    seed=args.random_state)
    study = openml.study.get_study(args.study_id, 'tasks')
    task_id = study.tasks[args.task_idx]
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()

    input_file = os.path.join(os.path.join(args.output_dir, str(task_id)), 'feature_importances.csv')
    importance_frame = pd.read_csv(input_file).sort_values('importance', ascending=False).reset_index(drop=True)

    for idx, record in importance_frame.iterrows():
        if idx >= args.feature_cutoff:
            break
        # Note: by looking at X, we assume some prior-knowledge
        non_integer_idx = [np.floor(val) == np.ceil(val) for _, val in enumerate(X[:, idx])]
        if not all(non_integer_idx):
            counter = collections.Counter(non_integer_idx)
            logging.info('Skipping feature %s, number of non-integers: %d/%d' % (record['name'],
                                                                                 counter[False],
                                                                                 X.shape[0]))
            continue

        all_indices = np.arange(args.feature_cutoff)
        numeric_indices = all_indices[np.arange(args.feature_cutoff) != idx]

        kwargs = {'n_iter': 20}
        clfs = {
            'numeric': sklearnbot.sklearn.as_search_cv(configuration_space, all_indices.tolist(), [], **kwargs),
            'nominal': sklearnbot.sklearn.as_search_cv(configuration_space, numeric_indices.tolist(), [idx], **kwargs)
        }

        for encoding, clf in clfs.items():
            clf.set_params(estimator__columntransformer__remainder='drop')
            output_dir = os.path.join(args.output_dir, str(task_id), record['name'], encoding, args.classifier_name)
            if os.path.isdir(output_dir):
                logging.warning('Directory already exists. Skipping. %s' % output_dir)
                continue
            run = openml.runs.run_model_on_task(task, clf)
            run.to_filesystem(output_dir, store_model=False)


if __name__ == '__main__':
    run(parse_args())
