from matplotlib import pyplot as plt
from absl import flags, app
import os
import pickle 
import pandas as pd


FLAGS = flags.FLAGS
flags.DEFINE_string("path", "./assets/results/", "Path to the results folder") 
flags.DEFINE_string("avg_uid", "", "Unique identifier for the experiment")
flags.DEFINE_string("client_uid", "", "Unique identifier for the experiment")
flags.DEFINE_string("com_uid", "", "Unique identifier for the experiment")

METHODS = ["fedavg", "client", "fedcompress"]

def find_files_by_uid(uid, path):
    files = os.listdir(path)
    target_files = [f for f in files if uid in f]
    metrics_file = [f for f in target_files if "metrics" in f][0]
    target_files.remove(metrics_file)
    arg_file = target_files[0]
    return arg_file, metrics_file

def eval_args(file_path: str, method: str, uid: str):
    args = pickle.load(open(file_path, "rb"))
    print(f"Arguments for {method} with uid {uid}:\n{args}\n")
    

def plot_metrics(avg_df, client_df, com_df, metrics_name,
                  title, x_label, y_label):
    plt.figure(figsize=(10, 5))
    plt.plot(avg_df[metrics_name], label="FedAvg", color="gray")
    plt.plot(client_df[metrics_name], label="Client Compression Only", color="blue")
    plt.plot(com_df[metrics_name], label="FedCompress", color="red")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.show()
    

def main(argv):
    del argv
    files = os.listdir(FLAGS.path)
    avg_arg_file, avg_metrics_file = find_files_by_uid(FLAGS.avg_uid, FLAGS.path)
    client_arg_file, client_metrics_file = find_files_by_uid(FLAGS.client_uid, FLAGS.path)
    com_arg_file, com_metrics_file = find_files_by_uid(FLAGS.com_uid, FLAGS.path)
    print(f"Files for fedavg: {avg_arg_file}, {avg_metrics_file}"
          f"\nFiles for client compression: {client_arg_file}, {client_metrics_file}"
          f"\nFiles for fedcompress: {com_arg_file}, {com_metrics_file}")
    eval_args(os.path.join(FLAGS.path, avg_arg_file), "FedAvg", FLAGS.avg_uid)
    eval_args(os.path.join(FLAGS.path, client_arg_file), "Client Compression", FLAGS.client_uid)
    eval_args(os.path.join(FLAGS.path, com_arg_file), "FedCompress", FLAGS.com_uid)
    
    # Convert metrics to pandas dataframe
    avg_df = pd.DataFrame(pickle.load(open(os.path.join(FLAGS.path, avg_metrics_file), "rb")))
    client_df = pd.DataFrame(pickle.load(open(os.path.join(FLAGS.path, client_metrics_file), "rb")))
    com_df = pd.DataFrame(pickle.load(open(os.path.join(FLAGS.path, com_metrics_file), "rb")))
    print(f"Metrics for FedAvg: shape-{avg_df.shape} \n{avg_df.columns}"
          f"\nMetrics for Client Compression: shape-{client_df.shape} \n{client_df.columns}"
          f"\nMetrics for FedCompress: shape-{com_df.shape} \n{com_df.columns}\n")
    
    # Plot metrics
    # plot_metrics(avg_df, client_df, com_df, 
    #              "test_accuracy", "Accuracy of Different Methods VS Epochs", "Epochs", "Accuracy (%)")
    # plot_metrics(avg_df, client_df, com_df, 
    #              "model_size", "Loss of Different Methods VS Epochs", "Epochs", "Model Size (KB)")
    plot_metrics(avg_df, client_df, com_df, 
                 "avg_comp_cost", "Avg Computational Cost VS Communication Rounds", "Rounds", "Energy (J)")
    plot_metrics(avg_df, client_df, com_df,
                 "avg_commu_cost", "Avg Communication Cost VS Communication Rounds", "Rounds", "Energy (J)")
    plot_metrics(avg_df, client_df, com_df,
                 "avg_total_cost", "Avg Total Cost VS Communication Rounds", "Rounds", "Energy (J)")
    plot_metrics(avg_df, client_df, com_df,
                 "cost_efficiency", "Cost Efficiency VS Communication Rounds", "Rounds", "Efficiecncy Rate")
    
    # print(f"Arguments for experiment {uid}: {args}\n\n")
    # df = pd.DataFrame(data)
    # print(f"Metrics for experiment {uid}: shape-{df.shape} \n{df.columns}")
    # print(df.head(5))
    


if __name__ == "__main__":
    app.run(main)

