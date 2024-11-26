import tensorflow as tf

# Define global variables
_regularizer = None
_shortcut_type = None
_dropout = None
_preact_projection = None
_cardinality = None
_bootleneck_width = None
_preact_shortcuts = None
_n_classes = None

class PadLayer(tf.keras.layers.Layer):
    def __init__(self, stride, filters, name=None, **kwargs):
        super(PadLayer, self).__init__(name=name, **kwargs)
        self.stride = stride
        self.filters = filters

    def call(self, x):
        return tf.pad(
            tf.keras.layers.MaxPool2D(1, self.stride)(x) if self.stride > 1 else x,
            paddings=[
                (0, 0),
                (0, 0),
                (0, 0),
                (0, self.filters - x.shape[-1]),
            ],
        )

    def get_config(self):
        config = super(PadLayer, self).get_config()
        config.update({"stride": self.stride, "filters": self.filters})
        return config

def early_exit_branch(x, num_classes, name_prefix):
    # Add a convolutional block
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        name=f"{name_prefix}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"{name_prefix}_pool1")(x)

    # Add the classifier
    x = tf.keras.layers.Flatten(name=f"{name_prefix}_flatten")(x)
    x = tf.keras.layers.Dense(
        num_classes, activation="softmax", name=f"{name_prefix}_output"
    )(x)

    return x

def regularized_padded_conv(*args, **kwargs):
    name = kwargs.pop("name", None)
    return tf.keras.layers.Conv2D(
        *args,
        **kwargs,
        padding="same",
        kernel_regularizer=_regularizer,
        kernel_initializer="he_normal",
        use_bias=False,
        name=name,
    )

def bn_relu(x, name_prefix):
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu")(x)
    return x

def shortcut(x, filters, stride, mode, name_prefix):
    if x.shape[-1] == filters:
        return x
    elif mode == "B":
        return regularized_padded_conv(
            filters, 1, strides=stride, name=f"{name_prefix}_conv"
        )(x)
    elif mode == "B_original":
        x = regularized_padded_conv(
            filters, 1, strides=stride, name=f"{name_prefix}_conv"
        )(x)
        return tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    elif mode == "A":
        return PadLayer(stride, filters, name=f"{name_prefix}_pad")(x)
    else:
        raise KeyError("Parameter shortcut_type not recognized!")

def original_block(x, filters, stride=1, name_prefix="", **kwargs):
    c1 = regularized_padded_conv(
        filters, 3, strides=stride, name=f"{name_prefix}_conv1"
    )(x)
    c2 = bn_relu(c1, name_prefix=f"{name_prefix}_bnrelu1")
    c2 = regularized_padded_conv(filters, 3, name=f"{name_prefix}_conv2")(c2)
    c2 = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(c2)
    mode = "B_original" if _shortcut_type == "B" else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode, name_prefix=f"{name_prefix}_shortcut")
    x = tf.keras.layers.Add(name=f"{name_prefix}_add")([x, c2])
    x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu")(x)
    return x

# Similarly, you can modify preactivation_block and bottleneck_block to include names
def preactivation_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)
    if preact_block:
        x = flow
    c1 = regularized_padded_conv(filters, 3, strides=stride)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)
    c2 = regularized_padded_conv(filters, 3)(bn_relu(c1))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c2

def bootleneck_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)
    if preact_block:
        x = flow
    c1 = regularized_padded_conv(filters//_bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(filters//_bootleneck_width, 3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(filters, 1)(bn_relu(c2))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c3

def group_of_blocks(
    x,
    block_type,
    num_blocks,
    filters,
    stride,
    block_idx=0,
    early_exit_points=None,
):
    global _preact_shortcuts, _n_classes
    preact_block = True if _preact_shortcuts or block_idx == 0 else False
    early_exists = []
    for i in range(num_blocks):
        block_name = f"block_{block_idx}_{i}"
        x = block_type(
            x,
            filters,
            stride if i == 0 else 1,
            preact_block=(preact_block if i == 0 else False),
            name_prefix=block_name,
        )

        # Add an early exit point
        if (block_idx, i) in early_exit_points:
            ee_name_prefix = f"early_exit_{block_idx}_{i}"
            early_output = early_exit_branch(x, _n_classes, name_prefix=ee_name_prefix)
            early_exists.append(early_output)

    return x, early_exists

def Resnet(
    input_shape,
    n_classes,
    l2_reg=1e-4,
    group_sizes=(2, 2, 2),
    features=(16, 32, 64),
    strides=(1, 2, 2),
    shortcut_type="B",
    block_type="preactivated",
    first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
    dropout=0,
    cardinality=1,
    bootleneck_width=4,
    preact_shortcuts=True,
    name="resnet",
    early_exit_points=None,
):
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, \
        _bootleneck_width, \
        _preact_shortcuts, \
        _n_classes
    
    _bootleneck_width = bootleneck_width
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type
    _cardinality = cardinality
    _dropout = dropout
    _preact_shortcuts = preact_shortcuts
    _n_classes = n_classes
    block_types = {
        "preactivated": preactivation_block,
        "bootleneck": bootleneck_block,
        "original": original_block,
    }

    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = regularized_padded_conv(**first_conv, name="initial_conv")(inputs)
    backbone_layers = [x]

    if block_type == "original":
        x = bn_relu(x, name_prefix="initial_bnrelu")
        backbone_layers.append(x)


    early_exists = None
    for block_idx, (group_size, feature, stride) in enumerate(
        zip(group_sizes, features, strides)
    ):
        x, exists = group_of_blocks(
            x,
            block_type=selected_block,
            num_blocks=group_size,
            block_idx=block_idx,
            filters=feature,
            stride=stride,
            early_exit_points=early_exit_points,
        )
        if exists:
            early_exists = exists
    

    # if block_type != "original":
    #     x = bn_relu(x, name_prefix="final_bnrelu")
    #     backbone_layers.append(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="GAP")(x)
    outputs = tf.keras.layers.Dense(
        n_classes, kernel_regularizer=_regularizer, name="output"
    )(x)

    # Combine all outputs
    model_outputs = early_exists + [outputs]

    model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name=name)
    # Collect early exit layers and backbone layers
    ee_layers = []
    backbone_layers = []
    ee_block_idx = early_exit_points[0][0] if early_exit_points else -1
    ee_block_inner_idx = early_exit_points[0][1] if early_exit_points else -1
    for layer in model.layers:
        if "input" in layer.name:
            ee_layers.append(layer)
        elif "initial" in layer.name:
            ee_layers.append(layer)
        elif "block" in layer.name and int(layer.name.split("_")[1]) < ee_block_idx:
            ee_layers.append(layer)
        elif "block" in layer.name and int(layer.name.split("_")[1]) == ee_block_idx and \
              int(layer.name.split("_")[2]) <= ee_block_inner_idx:
            ee_layers.append(layer)
        elif "early_exit" in layer.name:
            ee_layers.append(layer)
        else:
            backbone_layers.append(layer)
    assert len(set(ee_layers)) + len(set(backbone_layers)) == len(model.layers)
    assert len(ee_layers) == len(set(ee_layers))
    assert len(backbone_layers) == len(set(backbone_layers))
    return model, ee_layers, backbone_layers

def resnet20_ee(
    input_shape=(32, 32, 3),
    num_classes=10,
    weights_dir=None,
    block_type="original",
    shortcut_type="A",
    l2_reg=1e-4,
    name="resnet20",
    early_exit_points=None,
):
    assert early_exit_points is None or len(early_exit_points) == 1, (
        "Resnet20-ee only supports one early exit point!"
    )
    model, ee_layers, backbone_layers = Resnet(
        input_shape=input_shape,
        n_classes=num_classes,
        l2_reg=l2_reg,
        group_sizes=(3, 3, 3),
        features=(16, 32, 64),
        strides=(1, 2, 2),
        first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
        shortcut_type=shortcut_type,
        block_type=block_type,
        preact_shortcuts=False,
        name=name,
        early_exit_points=early_exit_points,
    )
    if weights_dir is not None:
        model.load_weights(weights_dir).expect_partial()
    return model, ee_layers, backbone_layers

############################################
# Define the model class
############################################

def calculate_entropy(hidden_state):
    """
    Calculate the entropy of the hidden state
    Args:
        hidden_state: tf.Tensor, the hidden state of the model
    Returns:
        entropy: tf.Tensor, the entropy of the hidden state
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    hidden_state = tf.clip_by_value(hidden_state, epsilon, 1.0)
    entropy = -tf.reduce_sum(hidden_state * tf.math.log(hidden_state), axis=1)
    return entropy

class EEModelWrapper(tf.keras.Model):
    def __init__(self, inputs, outputs, name, ee_layers, backbone_layers, ee_threshold=1.0):
        super(EEModelWrapper, self).__init__(inputs=inputs, outputs=outputs, name=name)
        self.ee_layers = ee_layers
        self.backbone_layers = backbone_layers
        self.ee_threshold = ee_threshold
        self.join_point_layer = self._get_joint_point()
        if self.join_point_layer is None:
            raise ValueError("No early exit point found in the model.")
        
        # Define metrics
        self.ee_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="ee_accuracy")
        self.bb_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="bb_accuracy")
        
    def _get_joint_point(self):
        joint_point = None
        for i, l in enumerate(self.ee_layers):
            if "early_exit" in l.name:
                joint_point = self.ee_layers[i - 1]
                break
        return joint_point

    def train_step(self, data):
        x_batch, y_batch = data
        with tf.GradientTape(persistent=True) as tape:
            ee_outputs, bb_outputs = self.call(x_batch, training=True)
            ee_loss = self.compiled_loss(y_batch, ee_outputs, regularization_losses=self.losses)
            bb_loss = self.compiled_loss(y_batch, bb_outputs, regularization_losses=self.losses)
            
        ee_trainable_vars = []
        bb_trainable_vars = []
        
        for layer in self.ee_layers:
            ee_trainable_vars += layer.trainable_variables
        for layer in self.backbone_layers:
            bb_trainable_vars += layer.trainable_variables

        gradients_ee = tape.gradient(ee_loss, ee_trainable_vars)
        gradients_bb = tape.gradient(bb_loss, bb_trainable_vars)
        # Apply gradients using different optimizers
        if isinstance(self.optimizer, (list, tuple)) and len(self.optimizer) == 2:
            self.optimizer[0].apply_gradients(zip(gradients_ee, ee_trainable_vars))
            self.optimizer[1].apply_gradients(zip(gradients_bb, bb_trainable_vars))
        else:
            raise ValueError("Two optimizers are required for training with early exits.")
        
        # Update metrics
        self.compiled_metrics.update_state(y_batch, ee_outputs)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({"ee_loss": ee_loss, "bb_loss": bb_loss})

        return metrics
    
    def test_step(self, data):
        x_batch, y_batch = data
        # FF through EE
        ee_output, Q = self.call_ee_layers(x_batch)
        entropy = calculate_entropy(ee_output)
        if tf.reduce_mean(entropy) < self.ee_threshold:
            self.ee_accuracy.update_state(y_batch, ee_output)
            self.accuracy.update_state(y_batch, ee_output)
            return {"accuracy": self.accuracy.result(), "ee_accuracy": self.ee_accuracy.result(), "bb_accuracy": self.bb_accuracy.result()}
        else:
            y_pred = self.call_backbone_layers(Q)
            self.bb_accuracy.update_state(y_batch, y_pred)
            self.accuracy.update_state(y_batch, y_pred)
            return {"accuracy": self.accuracy.result(), "ee_accuracy": self.ee_accuracy.result(), "bb_accuracy": self.bb_accuracy.result()}
    
    def call_ee_layers(self, x):
        """ Run the model layers to the ee output """
        join_state = None
        for layer in self.ee_layers:
            print(f"Layer: {layer}, x: {x.shape}")
            x = layer(x)
            if layer == self.join_point_layer:
                join_state = x
        return x, join_state
    
    def call_backbone_layers(self, x):
        """ Run the backbone layers from the join point to the end """
        for layer in self.backbone_layers:
            x = layer(x)
        return x
            
    def inference_with_ee(self, x):
        """ Run the model and return the ee output """
        ee_output, joint_point_state = self.call_ee_layers(x)
        entropy = calculate_entropy(joint_point_state)
        if tf.reduce_mean(entropy) < self.ee_threshold:
            return ee_output
        bb_output = self.call_backbone_layers(joint_point_state)
        return bb_output

    def call(self, inputs, training=False):
        if training:
            return self.resnet(inputs, training=training)
        else:
            return self.inference_with_ee(inputs)