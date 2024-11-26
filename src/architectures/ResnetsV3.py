import tensorflow as tf

class PadLayer(tf.keras.layers.Layer):
    def __init__(self, stride, filters, **kwargs):
        super(PadLayer, self).__init__(**kwargs)
        self.stride = stride
        self.filters = filters
        self.maxpool = tf.keras.layers.MaxPool2D(1, self.stride) if self.stride > 1 else None

    def call(self, x):
        if self.maxpool:
            x = self.maxpool(x)
        padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, self.filters - x.shape[-1]]])
        return tf.pad(x, padding)

    def get_config(self):
        config = super(PadLayer, self).get_config()
        config.update({
            "stride": self.stride,
            "filters": self.filters,
        })
        return config

def regularized_padded_conv(filters, kernel_size, strides=1, l2_reg=1e-4, name=None):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                  kernel_initializer='he_normal', use_bias=False, name=name)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, block_type='preactivated', shortcut_type='B',
                 l2_reg=1e-4, dropout=0, preact_block=False, block_idx=0, layer_idx=0, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.block_type = block_type
        self.shortcut_type = shortcut_type
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.preact_block = preact_block
        self.block_idx = block_idx
        self.layer_idx = layer_idx

        if block_type == 'original':
            self.c1 = regularized_padded_conv(filters, 3, strides=stride, l2_reg=self.l2_reg,
                                              name=f'block_{block_idx}_layer_{layer_idx}_conv1')
            self.bn1 = tf.keras.layers.BatchNormalization(name=f'block_{block_idx}_layer_{layer_idx}_bn1')
            self.relu1 = tf.keras.layers.ReLU(name=f'block_{block_idx}_layer_{layer_idx}_relu1')
            self.c2 = regularized_padded_conv(filters, 3, l2_reg=self.l2_reg,
                                              name=f'block_{block_idx}_layer_{layer_idx}_conv2')
            self.bn2 = tf.keras.layers.BatchNormalization(name=f'block_{block_idx}_layer_{layer_idx}_bn2')
            self.shortcut_layer = self._build_shortcut()
            self.final_relu = tf.keras.layers.ReLU(name=f'block_{block_idx}_layer_{layer_idx}_relu_out')
        elif block_type == 'preactivated':
            self.bn1 = tf.keras.layers.BatchNormalization(name=f'block_{block_idx}_layer_{layer_idx}_bn1')
            self.relu1 = tf.keras.layers.ReLU(name=f'block_{block_idx}_layer_{layer_idx}_relu1')
            self.c1 = regularized_padded_conv(filters, 3, strides=stride, l2_reg=self.l2_reg,
                                              name=f'block_{block_idx}_layer_{layer_idx}_conv1')
            if self.dropout:
                self.dropout_layer = tf.keras.layers.Dropout(self.dropout, name=f'block_{block_idx}_layer_{layer_idx}_dropout')
            else:
                self.dropout_layer = None
            self.bn2 = tf.keras.layers.BatchNormalization(name=f'block_{block_idx}_layer_{layer_idx}_bn2')
            self.relu2 = tf.keras.layers.ReLU(name=f'block_{block_idx}_layer_{layer_idx}_relu2')
            self.c2 = regularized_padded_conv(filters, 3, l2_reg=self.l2_reg,
                                              name=f'block_{block_idx}_layer_{layer_idx}_conv2')
            self.shortcut_layer = self._build_shortcut()
        else:
            raise ValueError("Unknown block_type: {}".format(block_type))

    def _build_shortcut(self):
        if self.shortcut_type == 'A':
            return PadLayer(self.stride, self.filters, name=f'block_{self.block_idx}_layer_{self.layer_idx}_shortcut_pad')
        elif self.shortcut_type == 'B':
            layers = [regularized_padded_conv(self.filters, 1, strides=self.stride, l2_reg=self.l2_reg,
                                              name=f'block_{self.block_idx}_layer_{self.layer_idx}_shortcut_conv')]
            if self.block_type == 'original':
                layers.append(tf.keras.layers.BatchNormalization(name=f'block_{self.block_idx}_layer_{self.layer_idx}_shortcut_bn'))
            return tf.keras.Sequential(layers, name=f'block_{self.block_idx}_layer_{self.layer_idx}_shortcut')
        else:
            raise KeyError("Parameter shortcut_type not recognized!")

    def call(self, x, training=None):
        if self.block_type == 'original':
            c1 = self.c1(x)
            c1 = self.bn1(c1, training=training)
            c1 = self.relu1(c1)
            c2 = self.c2(c1)
            c2 = self.bn2(c2, training=training)
            shortcut = self.shortcut_layer(x, training=training)
            x = tf.keras.layers.Add(name=f'block_{self.block_idx}_layer_{self.layer_idx}_add')([shortcut, c2])
            x = self.final_relu(x)
        elif self.block_type == 'preactivated':
            flow = self.bn1(x, training=training)
            flow = self.relu1(flow)
            if self.preact_block:
                x = flow
            c1 = self.c1(flow)
            if self.dropout_layer:
                c1 = self.dropout_layer(c1, training=training)
            c1 = self.bn2(c1, training=training)
            c1 = self.relu2(c1)
            c2 = self.c2(c1)
            shortcut = self.shortcut_layer(x)
            x = tf.keras.layers.Add(name=f'block_{self.block_idx}_layer_{self.layer_idx}_add')([shortcut, c2])
        else:
            raise NotImplementedError("Block type '{}' not implemented.".format(self.block_type))
        return x
    

class ResNet(tf.keras.Model):
    def __init__(self, input_shape, n_classes, l2_reg=1e-4, group_sizes=(2,2,2),
                 features=(16,32,64), strides=(1,2,2), shortcut_type='B',
                 block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                 dropout=0, preact_shortcuts=True, name='resnet'):
        super(ResNet, self).__init__(name=name)
        self.l2_reg = l2_reg
        self.group_sizes = group_sizes
        self.features = features
        self.strides = strides
        self.shortcut_type = shortcut_type
        self.block_type = block_type
        self.first_conv_params = first_conv
        self.dropout = dropout
        self.preact_shortcuts = preact_shortcuts

        # Initial convolution layer
        self.first_conv = regularized_padded_conv(**self.first_conv_params, l2_reg=self.l2_reg, name='conv_initial')
        if self.block_type == 'original':
            self.bn1 = tf.keras.layers.BatchNormalization(name='bn_initial')
            self.relu1 = tf.keras.layers.ReLU(name='relu_initial')

        # Build residual blocks
        self.blocks = []
        for block_idx, (group_size, feature, stride) in enumerate(zip(self.group_sizes, self.features, self.strides)):
            group = self.group_of_blocks(group_size, feature, stride, block_idx)
            self.blocks.append(group)

        # Final batch norm and ReLU
        if self.block_type != 'original':
            self.final_bn = tf.keras.layers.BatchNormalization(name='bn_final')
            self.final_relu = tf.keras.layers.ReLU(name='relu_final')

        # Global average pooling and output layer
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')
        self.fc = tf.keras.layers.Dense(n_classes, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='fc')

    def group_of_blocks(self, num_blocks, filters, stride, block_idx):
        layers = []
        preact_block = self.preact_shortcuts or block_idx == 0
        layers.append(ResidualBlock(filters, stride=stride, block_type=self.block_type,
                                    shortcut_type=self.shortcut_type, l2_reg=self.l2_reg,
                                    dropout=self.dropout, preact_block=preact_block,
                                    block_idx=block_idx, layer_idx=0))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(filters, stride=1, block_type=self.block_type,
                                        shortcut_type=self.shortcut_type, l2_reg=self.l2_reg,
                                        dropout=self.dropout, preact_block=False,
                                        block_idx=block_idx, layer_idx=i))
        return tf.keras.Sequential(layers, name=f'group_{block_idx}')

    def call(self, inputs, training=None):
        x = self.first_conv(inputs)
        if self.block_type == 'original':
            x = self.bn1(x, training=training)
            x = self.relu1(x)
        for block in self.blocks:
            x = block(x, training=training)
        if self.block_type != 'original':
            x = self.final_bn(x, training=training)
            x = self.final_relu(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def resnet20(cls, input_shape=(32,32,3), num_classes=10, weights_dir=None,
                 block_type='original', shortcut_type='A', l2_reg=1e-4, name="resnet20"):
        inputs = tf.keras.layers.Input(shape=input_shape, name='input')
        x = inputs
        model = cls(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, group_sizes=(3, 3, 3),
                    features=(16, 32, 64), strides=(1, 2, 2),
                    first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type,
                    block_type=block_type, preact_shortcuts=False, name=name)
        outputs = model(x)
        keras_model = tf.keras.Model(inputs, outputs, name=name)
        if weights_dir is not None:
            model.load_weights(weights_dir).expect_partial()
        return keras_model

from tensorflow.keras.utils import plot_model
model = ResNet.resnet20()
# x = tf.random.normal((1, 32, 32, 3))
# _ = model(x)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)