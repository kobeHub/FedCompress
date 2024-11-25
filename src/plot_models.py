from absl import app, flags
from network import get_resnet20_network
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "resnet20", "resnet20 or resnet20-ee")
flags.DEFINE_string("option", "keras", "keras or tf")
flags.DEFINE_list("early_exit_points", [(0, 0), (1, 0), (2, 0)], "List of early exit points.")
flags.DEFINE_string("out_dir", "./assets/imgs", "Output file directory.")
flags.DEFINE_string("tflog_dir", "./assets/tflogs", "Output log directory.")


def main(argv):
    del argv  # Unused.
    model_name = FLAGS.model
    if model_name == "resnet20":
        model = get_resnet20_network(early_exit_points=None)
    elif model_name == "resnet20-ee":
        assert FLAGS.early_exit_points is not None
        print(f"Early exit points: {FLAGS.early_exit_points}")
        model = get_resnet20_network(early_exit_points=FLAGS.early_exit_points)

    # model.summary()
    if FLAGS.option == "keras":
        plot_model(model, to_file=f"{FLAGS.out_dir}/{model_name}-{FLAGS.option}.png",
                    show_shapes=True, show_layer_names=True)
    elif FLAGS.option == "tf":
        log_path = FLAGS.tflog_dir
        os.makedirs(log_path, exist_ok=True)   
        writer = tf.summary.create_file_writer(log_path)
        dummy_input = tf.random.normal((1, 32, 32, 3))
        tf.summary.trace_on(graph=True, profiler=False)
        model(dummy_input)
        with writer.as_default():
            tf.summary.trace_export(name=model_name, step=0)


if __name__ == '__main__':
    app.run(main)