import argparse
import feature_encoding
import openml
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--task_idx', type=int, default=18)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/feature_encoding')
    parser.add_argument('--n_trees', type=int, default=16)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()


def run(args):
    study = openml.study.get_study(args.study_id, 'tasks')
    task_id = study.tasks[args.task_idx]
    # TODO: make this lazy, see issue in issue tracker.
    task = openml.tasks.get_task(task_id)
    df = feature_encoding.feature_importance_on_openml_task(task, args.n_trees, args.random_state)

    output_dir = os.path.join(args.output_dir, str(task_id))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'feature_importances.csv'))


if __name__ == '__main__':
    run(parse_args())
