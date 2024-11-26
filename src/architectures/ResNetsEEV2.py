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
    

class ResNetEE(tf.keras.Model):
    def __init__(self, input_shape, n_classes, l2_reg=1e-4, group_sizes=(2,2,2),
                 features=(16,32,64), strides=(1,2,2), shortcut_type='B',
                 block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                 dropout=0, preact_shortcuts=True, name='resnetEE', ee_location=(1, 0), ee_threshold=0.1):
        super(ResNetEE, self).__init__(name=name)
        self.l2_reg = l2_reg
        self.group_sizes = group_sizes
        self.features = features
        self.strides = strides
        self.shortcut_type = shortcut_type
        self.block_type = block_type
        self.first_conv_params = first_conv
        self.dropout = dropout
        self.preact_shortcuts = preact_shortcuts
        self.ee_location = ee_location
        self.ee_threshold = ee_threshold

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

        # Define early exit layers
        if self.ee_location is not None:
            self.ee_branch = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                       name='ee_conv'),
                tf.keras.layers.GlobalAveragePooling2D(name='ee_global_pool'),
                tf.keras.layers.Dense(n_classes, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                      name='ee_output')
            ], name='ee_branch')

        # Global average pooling and output layer
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='GMP_layer')
        self.fc = tf.keras.layers.Dense(n_classes, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='main_output')

    def build(self, input_shape):
        super(ResNetEE, self).build(input_shape)
        # Record the ee layers and backbone layers
        # Partition layers into common_layers, ee_layers, and bb_layers
        self.common_layers = []
        self.ee_layers = []
        self.bb_layers = []

        ee_block_idx, ee_layer_idx = self.ee_location

        # Collect common layers
        self.common_layers.append(self.first_conv)
        if self.block_type == 'original':
            self.common_layers.extend([self.bn1, self.relu1])

        # Collect layers up to EE point
        for block_idx, block in enumerate(self.blocks):
            if block_idx < ee_block_idx:
                self.common_layers.append(block)
            elif block_idx == ee_block_idx:
                # Split within the block
                common_sub_layers = []
                bb_sub_layers = []
                common_layer_idxs = []
                bb_layer_idx = []
                for layer in block.layers:
                    if layer.layer_idx <= ee_layer_idx:
                        common_layer_idxs.append(str(layer.layer_idx))
                        common_sub_layers.append(layer)
                    else:
                        bb_layer_idx.append(str(layer.layer_idx))
                        bb_sub_layers.append(layer)
                # Reassign blocks
                common_suffix = '_'.join(common_layer_idxs)
                bb_suffix = '_'.join(bb_layer_idx)
                common_block = tf.keras.Sequential(common_sub_layers, name=f'group_{block_idx}_common_layers_{common_suffix}')
                bb_block = tf.keras.Sequential(bb_sub_layers, name=f'group_{block_idx}_bb_layers_{bb_suffix}')
                self.common_layers.append(common_block)
                self.bb_layers.append(bb_block)
            else:
                self.bb_layers.append(block)
        # Include EE branch
        self.ee_layers.append(self.ee_branch)
        # Include BB final layers
        if self.block_type != 'original':
            self.bb_layers.extend([self.final_bn, self.final_relu])
        self.bb_layers.extend([self.global_pool, self.fc])
        print("Common layers:")
        for layer in self.common_layers:
            print(f"  {layer.name}")
        print("Early exit layers:")
        for layer in self.ee_layers:
            print(f"  {layer.name}")
        print("Backbone layers:")
        for layer in self.bb_layers:
            print(f"  {layer.name}")
            

    def collect_trainable_vars(self):
         # Collect trainable variables
        common_trainable_vars = [var for layer in self.common_layers for var in layer.trainable_variables]
        ee_trainable_vars = [var for layer in self.ee_layers for var in layer.trainable_variables]
        bb_trainable_vars = [var for layer in self.bb_layers for var in layer.trainable_variables]
        return common_trainable_vars, ee_trainable_vars, bb_trainable_vars
        

    def group_of_blocks(self, num_blocks, filters, stride, block_idx):
        layers = []
        preact_block = self.preact_shortcuts or block_idx == 0
        layers.append(ResidualBlock(filters, stride=stride, block_type=self.block_type,
                                    shortcut_type=self.shortcut_type, l2_reg=self.l2_reg,
                                    dropout=self.dropout, preact_block=preact_block,
                                    block_idx=block_idx, layer_idx=0, name=f"group_{block_idx}_block_0"))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(filters, stride=1, block_type=self.block_type,
                                        shortcut_type=self.shortcut_type, l2_reg=self.l2_reg,
                                        dropout=self.dropout, preact_block=False,
                                        block_idx=block_idx, layer_idx=i, name=f"group_{block_idx}_block_{i}"))
        return tf.keras.Sequential(layers, name=f'group_{block_idx}')

    def call(self, inputs, training=None):
        # Process common layers
        x = inputs
        for layer in self.common_layers:
            x = layer(x, training=training)

        # Compute EE output
        ee_x = x
        for layer in self.ee_branch.layers:
            ee_x = layer(ee_x, training=training)
        ee_output = ee_x  # Shape: (batch_size, num_classes)

        # Compute entropy
        entropy = self.compute_entropy(ee_output)

        def exit_early():
            return ee_output

        def continue_forward():
            # Process backbone layers
            x_bb = x
            for layer in self.bb_layers:
                x_bb = layer(x_bb, training=training)
            return x_bb

        # Decide whether to exit early
        if not training:
            # During inference, make the decision
            entropy_threshold = self.ee_threshold
            use_ee = tf.less(entropy, entropy_threshold)
            use_ee = tf.cast(use_ee, tf.bool)
            # For each sample in the batch, decide whether to exit early
            final_output = tf.where(tf.expand_dims(use_ee, axis=-1), ee_output, continue_forward())
            return final_output
        else:
            # During training, compute both outputs
            # Process backbone layers
            x_bb = x
            for layer in self.bb_layers:
                x_bb = layer(x_bb, training=training)
            main_output = x_bb
            return ee_output, main_output


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            ee_output, main_output = self(x, training=True)
            main_loss = self.compiled_loss(y, main_output, regularization_losses=self.losses)
            ee_loss = self.compiled_loss(y, ee_output, regularization_losses=self.losses)

        common_vars, ee_vars, bb_vars = self.collect_trainable_vars()
        ee_vars = common_vars + ee_vars
        main_gradients = tape.gradient(main_loss, bb_vars)
        ee_gradients = tape.gradient(ee_loss, ee_vars)
        self.optimizer.apply_gradients(zip(main_gradients, bb_vars))
        self.optimizer.apply_gradients(zip(ee_gradients, ee_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, main_output)
        self.compiled_metrics.update_state(y, ee_output)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['main_loss'] = main_loss
        metrics['ee_loss'] = ee_loss 
         
        return metrics

    def test_step(self, data):
        x, y = data
        final_output = self(x, training=False)
        # Compute loss
        self.compiled_loss(y, final_output, regularization_losses=self.losses)
        # Update metrics
        self.compiled_metrics.update_state(y, final_output)
        return {m.name: m.result() for m in self.metrics}
    
    @classmethod
    def compute_entropy(cls, logits):
        probs = tf.nn.softmax(logits, axis=-1)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        return entropy

    @classmethod
    def resnet20_ee(cls, input_shape=(32,32,3), num_classes=10, weights_dir=None,
                 block_type='original', shortcut_type='A', l2_reg=1e-4, name="resnet20-ee", 
                 ee_location=(1, 0), ee_threshold=0.1):
        inputs = tf.keras.layers.Input(shape=input_shape, name='input')
        x = inputs
        model = cls(input_shape=input_shape, n_classes=num_classes, l2_reg=l2_reg, group_sizes=(3, 3, 3),
                    features=(16, 32, 64), strides=(1, 2, 2),
                    first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type,
                    block_type=block_type, preact_shortcuts=False, name=name, 
                    ee_location=ee_location, ee_threshold=ee_threshold)
        outputs = model(x)
        keras_model = tf.keras.Model(inputs, outputs, name=name)
        if weights_dir is not None:
            model.load_weights(weights_dir).expect_partial()
        return keras_model
