import argparse
import copy
import logging
import os
import openmlcontrib
import sklearn


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    input_dir = os.path.join(os.path.expanduser('~'), 'habanero_experiments/feature_encoding')
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_task_limit', type=int, default=10)
    parser.add_argument('--task_limit', type=int, default=None)
    parser.add_argument('--classifier_name', type=str, default='decision_tree')
    parser.add_argument('--input_dir', type=str, default=input_dir)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    df = openmlcontrib.misc.results_from_folder_to_df(args.input_dir, sklearn.metrics.accuracy_score)
    df = df.rename(index=str, columns={
        "folder_depth_0": "task_id",
        "folder_depth_1": "feature_name",
        "folder_depth_2": "feature_strategy",
        "folder_depth_3": "classifier"
    })
    df = df.loc[df['classifier'] == args.classifier_name]
    del df['classifier']
    df = df.set_index(['task_id', 'feature_name', 'feature_strategy'])
    results = []
    for idx, row_nom in df.iterrows():
        if idx[1] != 'nominal':
            continue
        idx_copy = (idx[0], idx[1], 'numeric')
        row_num = df.loc[idx_copy]
        results.append(abs(row_nom['y'] - row_num['y']))
    print(results)
    print(max(results))


if __name__ == '__main__':
    run(parse_args())
