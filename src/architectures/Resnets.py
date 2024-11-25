import tensorflow as tf

class PadLayer(tf.keras.layers.Layer):
    def __init__(self, stride, filters, **kwargs):
        super(PadLayer, self).__init__(**kwargs)
        self.stride = stride
        self.filters = filters

    def call(self, x):
        return tf.pad(tf.keras.layers.MaxPool2D(1, self.stride)(x) if self.stride>1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, self.filters - x.shape[-1])])
        
    def get_config(self):
        # Add stride and filters to the config dictionary
        config = super(PadLayer, self).get_config()
        config.update({
            "stride": self.stride,
            "filters": self.filters,
        })
        return config

def early_exit_branch(x, num_classes, name=None):
    # Add a convolutional block
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    
    # # Add the classifier
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dense(num_classes, activation='softmax', name=name)(x)
    # x = tf.keras.layers.GlobalAveragePooling2D(name=f'GMP_layer_{name}')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name=name)(x)

    return x

def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer,
        kernel_initializer='he_normal', use_bias=False)

def bn_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    elif mode == 'B':
        return regularized_padded_conv(filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(filters, 1, strides=stride)(x)
        return tf.keras.layers.BatchNormalization()(x)
    elif mode == 'A':
        # return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
        #     paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
        return PadLayer(stride, filters)(x)
    else:
        raise KeyError("Parameter shortcut_type not recognized!")

def original_block(x, filters, stride=1, **kwargs):
    c1 = regularized_padded_conv(filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, 3)(bn_relu(c1))
    c2 = tf.keras.layers.BatchNormalization()(c2)
    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode)
    x = tf.keras.layers.Add()([x, c2])
    return tf.keras.layers.ReLU()(x)

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

def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0, early_exit_points=None):
    global _preact_shortcuts, _n_classes
    preact_block = True if _preact_shortcuts or block_idx == 0 else False
    early_exists = []
    for i in range(num_blocks):
        x = block_type(x, filters, stride if i == 0 else 1, preact_block=(preact_block if i == 0 else False)) 
        
        # Add an early exit point
        if early_exit_points is not None and (block_idx, i) in early_exit_points:
            early_output = early_exit_branch(x, _n_classes, name=f'early_exit_{block_idx}_{i}')
            early_exists.append(early_output)
    return x, early_exists

def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
        shortcut_type='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
        dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True, name='resnet', early_exit_points=None):

    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts, _n_classes
    _bootleneck_width = bootleneck_width
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type
    _cardinality = cardinality
    _dropout = dropout
    _preact_shortcuts = preact_shortcuts
    _n_classes = n_classes
    block_types = {'preactivated': preactivation_block, 'bootleneck': bootleneck_block, 'original': original_block}

    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)

    if block_type == 'original':
        flow = bn_relu(flow)
        
    # To store early exit points
    early_exists = []

    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        flow, exists = group_of_blocks(flow, block_type=selected_block,
                    num_blocks=group_size, block_idx=block_idx,
                    filters=feature, stride=stride, early_exit_points=early_exit_points)
        early_exists.extend(exists)
        
    if block_type != 'original':
        flow = bn_relu(flow)
    flow = tf.keras.layers.GlobalAveragePooling2D(name='GMP_layer')(flow)
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    
    # Combine all outputs
    model_outputs = early_exists + [outputs] if early_exit_points is not None else outputs

    model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name=name)
    return model

def resnet20(input_shape=(32,32,3), num_classes=10, weights_dir=None,
    	block_type='original', shortcut_type='A', l2_reg=1e-4, name="resnet20", early_exit_points=None):
	model = Resnet(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, group_sizes=(3, 3, 3), features=(16, 32, 64),
		strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type,
		block_type=block_type, preact_shortcuts=False, name=name, early_exit_points=early_exit_points)
	if weights_dir is not None: model.load_weights(weights_dir).expect_partial()
	return model