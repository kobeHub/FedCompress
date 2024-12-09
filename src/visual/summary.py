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
flags.DEFINE_bool("iid", True, "IID data or not")
flags.DEFINE_string("save_dir", "./assets/res-new/", "Path to save the images")


METHODS = ["fedavg", "client", "fedcompress"]
EXPER_UIDS = {"iid": {
    "fedavg": "12081881e6",
    "client": "a70b0ec677",
    "fedcompress": "af03e5758d"
},
"noniid": {
    "fedavg": "52d832652d",
    "client": "b0be5185ae",
    "fedcompress": "9ef0d94f07"
}}
FONT_SIZE = 18


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
    
def plot_metrics(avg_df, client_df, com_df, metrics_name, suffix,
                  title, x_label, y_label, y_lim=None, res_dir="./assets/res-imgs/"):
    plt.figure(figsize=(10, 5))
    if y_lim:
        plt.ylim(y_lim)
    if metrics_name in avg_df:
        plt.plot(avg_df[metrics_name], label="FedAvg", color="gray")
    plt.plot(client_df[metrics_name], label="Client Compression Only", color="blue")
    plt.plot(com_df[metrics_name], label="FedCompress", color="red")
    # plt.title(title)
    plt.xticks(fontsize=FONT_SIZE - 2)
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.grid()
    plt.legend(fontsize=FONT_SIZE)
    plt.tight_layout(pad=0.1)
    fig_path = os.path.join(res_dir, f"{metrics_name}-{suffix}.png")
    print(f"Saving figure to {fig_path}")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    
def eval_metrics(last_round, name, max_size):
    compressed_ratio = (max_size - last_round["model_size"].values[0]) / max_size
    print(f"{name}: Accuracy-{last_round['test_accuracy'].values[0]:.4f}, ",
           f"Score-{last_round['val_score'].values[0]:.4f}, "
        f"Model Size-{last_round['model_size'].values[0]:.4f}, Score-{last_round['val_score'].values[0]:.4f}, "
        f"Cost Efficiency-{last_round['cost_efficiency'].values[0]:.4f}"
        f"Compression ratio-{compressed_ratio:.4f}")
    
def main(argv):
    del argv
    if FLAGS.iid:
        avg_uid = EXPER_UIDS["iid"]["fedavg"]
        client_uid = EXPER_UIDS["iid"]["client"]
        com_uid = EXPER_UIDS["iid"]["fedcompress"]
    else:
        avg_uid = EXPER_UIDS["noniid"]["fedavg"]
        client_uid = EXPER_UIDS["noniid"]["client"]
        com_uid = EXPER_UIDS["noniid"]["fedcompress"]
    avg_arg_file, avg_metrics_file = find_files_by_uid(avg_uid, FLAGS.path)
    client_arg_file, client_metrics_file = find_files_by_uid(client_uid, FLAGS.path)
    com_arg_file, com_metrics_file = find_files_by_uid(com_uid, FLAGS.path)
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
    print(f"Columns for FedAvg: {avg_df.columns}\nColumns for Client: {client_df.columns}\nColumns for FedCompress: {com_df.columns}")

    res_dir = FLAGS.save_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)

    # Eval the results of the last rounds 
    last_round = avg_df["rnd"].max()
    last_avg = avg_df[avg_df["rnd"] == last_round]
    last_client = client_df[client_df["rnd"] == last_round]
    last_com = com_df[com_df["rnd"] == last_round]
    max_size = last_avg["model_size"].max()
    eval_metrics(last_avg, "Fedavg", max_size)
    eval_metrics(last_client, "Client", max_size)
    eval_metrics(last_com, "FedCompress", max_size)
    

    return

    # Plot metrics
    suffix = "IID" if FLAGS.iid else "Non-IID"
    plot_metrics(avg_df, client_df, com_df,
                 "num_clusters", suffix, f"Number of Clusters VS Communication Rounds on {suffix} Data",
                   "Rounds", "Cluster Count", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "val_score", suffix, f"Quality Score VS Communciation Rounds on {suffix} Data",
                   "Rounds", "Score", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df, 
                 "test_accuracy", suffix, f"Accuracy VS Communication Rounds on {suffix} Data",
                   "Epochs", "Accuracy (%)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df, 
                 "model_size", suffix, f"Model Size VS Communication Rounds on {suffix} Data", 
                 "Epochs", "Model Size (KB)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df, 
                 "avg_comp_cost", suffix, f"Avg Computational Cost VS Communication Rounds on {suffix} Data",
                   "Rounds", "Energy (J)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "avg_commu_cost", suffix, f"Avg Communication Cost VS Communication Rounds on {suffix} Data",
                   "Rounds", "Energy (J)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "cumulative_comp_costs", suffix, f"Cumulative Computation Cost VS Communication Rounds on {suffix} Data",
                   "Rounds", "Energy (J)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "cumulative_commu_costs", suffix, f"Cumulative Communication Cost VS Communication Rounds on {suffix} Data",
                   "Rounds", "Energy (J)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "avg_total_cost", suffix, f"Avg Total Cost VS Communication Rounds on {suffix} Data",
                   "Rounds", "Energy (J)", y_lim=None, res_dir=res_dir)
    plot_metrics(avg_df, client_df, com_df,
                 "cost_efficiency", suffix, f"Cost Efficiency VS Communication Rounds on {suffix} Data",
                   "Rounds", "Efficiecncy Rate", y_lim=None, res_dir=res_dir)
 
    # print(f"Arguments for experiment {uid}: {args}\n\n")
    # df = pd.DataFrame(data)
    # print(f"Metrics for experiment {uid}: shape-{df.shape} \n{df.columns}")
    # print(df.head(5))
    
if __name__ == "__main__":
    app.run(main)