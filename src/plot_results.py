import pickle
from absl import app, flags
import os
import numpy as np 
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string("option", "args", "Eval args or metrics.")
flags.DEFINE_string('results_dir', './assets/results', 'Directory to save results.')
flags.DEFINE_string('method', 'fedavg', 'Method to use. One of `fedavg`, `fedcompress`, `client`.')

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def main(argv):
    del argv  # Unused.
    results_dir = FLAGS.results_dir
    option = FLAGS.option
    file_list = os.listdir(results_dir)
    target_file = [f for f in file_list if f.endswith('.pkl') 
                    and FLAGS.method in f and option in f][0]
    target_path = os.path.join(results_dir, target_file)
    print(f"Loading {option} from {target_path}")
    with open(target_path, 'rb') as f:
        data = pickle.load(f)
    if option == 'args':
        print(f"Args: {data}")
        return
    df = pd.DataFrame(data)
    print(f"Columns: {df.columns}")
    print(f"Metrics: {df['cost_efficiency']}]")

if __name__ == '__main__':
    app.run(main)


