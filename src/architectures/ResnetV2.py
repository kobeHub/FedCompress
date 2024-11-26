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
    
    
class BatchEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, threshold=0.8, name='batch_entropy', **kwargs):
        super(BatchEntropyLayer, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        
    def call(self, inputs):
        # Apply softmax to get probabilities
        probs = tf.nn.softmax(inputs, axis=-1)
        
        # Avoid numerical instability
        epsilon = 1e-10
        probs = tf.clip_by_value(probs, epsilon, 1.0)
        
        # Compute entropy per sample in batch
        batch_entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=1)
        
        # Return boolean mask based on threshold
        return tf.less(batch_entropy, self.threshold)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
        
    def get_config(self):
        config = super(BatchEntropyLayer, self).get_config()
        config.update({"threshold": self.threshold})
        return config


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

# def compute_entropy(hidden_state):
#     epsilon = 1e-10
#     hidden_state = tf.clip_by_value(hidden_state, epsilon, 1.0)
#     entropy = -tf.reduce_sum(hidden_state * tf.math.log(hidden_state), axis=1)
#     return entropy

class ResNet20(tf.keras.Model):
    def __init__(
        self,
        input_shape=(32, 32, 3),
        num_classes=10,
        l2_reg=1e-4,
        block_type='original',
        shortcut_type='A',
        name="resnet20",
        ee_location=(1, 0),
        ee_threshold=0.,
        **kwargs
    ):
        super(ResNet20, self).__init__(name=name, **kwargs)

        # Initialize model parameters
        self.input_shape = input_shape
        self.l2_reg = l2_reg
        self.regularizer = tf.keras.regularizers.l2(l2_reg)
        self.shortcut_type = shortcut_type
        self.block_type = block_type
        self.num_classes = num_classes
        self.ee_location = ee_location
        self.ee_threshold = ee_threshold
        
        # Model architecture parameters
        self.group_sizes = (3, 3, 3)
        self.features = (16, 32, 64)
        self.strides = (1, 2, 2)
        self.first_conv = {"filters": 16, "kernel_size": 3, "strides": 1}
        
        # Metrics
        self.ee_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='ee_accuracy')
        self.main_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='main_accuracy')
        self.early_exit_rate = tf.keras.metrics.Mean(name='early_exit_rate')

        # Build the model
        self.build_model()
        self.ee_layers, self.backbone_layers = self._divide_model_layers()
        print(f"Early exit layers:")
        for layer in self.ee_layers:
            print(f"  {layer.name}")
        print(f"Backbone layers:")
        for layer in self.backbone_layers:
            print(f"  {layer.name}")
        
            

        # Separate optimizers
        self.ee_optimizer = None
        self.backbone_optimizer = None

    def compile(self, ee_optimizer='adam', backbone_optimizer='adam', **kwargs):
        """Custom compile with separate optimizers"""
        super().compile(**kwargs)
        
        # Create optimizers if strings provided
        if isinstance(ee_optimizer, str):
            self.ee_optimizer = tf.keras.optimizers.get(ee_optimizer)
        else:
            self.ee_optimizer = ee_optimizer
            
        if isinstance(backbone_optimizer, str):
            self.backbone_optimizer = tf.keras.optimizers.get(backbone_optimizer)
        else:
            self.backbone_optimizer = backbone_optimizer


    def early_exit_branch(self, x):
        x = self.ee_conv1(x)
        x = self.ee_bn(x)
        x = self.ee_relu(x)
        x = self.ee_pool(x)
        x = self.ee_flatten(x)
        return self.ee_dense(x)
    
    def _divide_model_layers(self):
        """Helper method to divide layers into EE and backbone groups"""
        ee_block_idx, ee_block_inner_idx = self.ee_location
        ee_layers = []
        backbone_layers = []

        for layer in self.layers:
            name_parts = layer.name.split("_")
            is_ee_layer = any([
                "input" in layer.name,
                "initial" in layer.name,
                "block" in layer.name and (
                    int(name_parts[2]) < ee_block_idx or
                    (int(name_parts[2]) == ee_block_idx and 
                     int(name_parts[3]) <= ee_block_inner_idx)
                ),
                "early_exit" in layer.name or "ee_" in layer.name,
            ])
            
            if is_ee_layer:
                ee_layers.append(layer)
            else:
                backbone_layers.append(layer)
                
        return ee_layers, backbone_layers
    
    def _get_ee_variables(self):
        """Get variables for early exit branch"""
        ee_vars = []
        for layer in self.ee_layers:
            ee_vars.extend(layer.trainable_variables)
        return ee_vars

    def _get_backbone_variables(self):
        """Get variables for backbone network"""
        backbone_vars = []
        for layer in self.backbone_layers:
            backbone_vars.extend(layer.trainable_variables)
        return backbone_vars
    
    def call(self, inputs, training=None):
        # 1. Initial layers
        x = self.initial_conv(inputs)
        if self.block_type == 'original':
            x = self.initial_bn(x)
            x = self.initial_relu(x)
        
        # 2. Pre-exit blocks
        join_state = None
        for block_idx, block_group in enumerate(self.pre_exit_blocks):
            for i, block in enumerate(block_group):
                identity = x
                x = block['conv1'](x)
                x = block['bn1'](x, training=training)
                x = block['relu1'](x)
                x = block['conv2'](x)
                x = block['bn2'](x, training=training)
                shortcut = self.shortcut(identity, x.shape[-1], block['conv1'].strides[0], f'pre_block_{block_idx}_{i}')
                x = block['add']([shortcut, x])
                x = block['relu'](x)
                
                if block['relu'] == self.join_point:
                    join_state = x
                    break
        
        # 3. Early exit branch
        ee_output = self.ee_conv1(join_state)
        ee_output = self.ee_bn(ee_output)
        ee_output = self.ee_relu(ee_output)
        ee_output = self.ee_pool(ee_output)
        ee_output = self.ee_flatten(ee_output)
        ee_output = self.ee_dense(ee_output)
        
        entropy = BatchEntropyLayer(threshold=self.ee_threshold)(ee_output)
        if tf.reduce_any(entropy):
            return ee_output, None
        
        # 4. Post-exit blocks
        x = join_state
        for block_idx, block_group in enumerate(self.post_exit_blocks):
            for i, block in enumerate(block_group):
                identity = x
                x = block['conv1'](x)
                x = block['bn1'](x, training=training)
                x = block['relu1'](x)
                x = block['conv2'](x)
                x = block['bn2'](x, training=training)
                shortcut = self.shortcut(identity, x.shape[-1], block['conv1'].strides[0], f'post_block_{block_idx}_{i}')
                x = block['add']([shortcut, x])
                x = block['relu'](x)
        
        # 5. Final layers
        x = self.global_pool(x)
        main_output = self.final_dense(x)
        
        return ee_output, main_output
                
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            ee_output, main_output = self(x, training=True)
            
            # Calculate losses
            ee_loss = self.compiled_loss(y, ee_output, regularization_losses=self.losses)
            main_loss = self.compiled_loss(y, main_output, regularization_losses=self.losses)
            total_loss = ee_loss + main_loss
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.ee_accuracy.update_state(y, ee_output)
        self.main_accuracy.update_state(y, main_output)
        
        return {
            'loss': total_loss,
            'ee_loss': ee_loss,
            'main_loss': main_loss,
            'ee_accuracy': self.ee_accuracy.result(),
            'main_accuracy': self.main_accuracy.result()
        }

    def test_step(self, data):
        x, y = data
        ee_output, main_output = self(x, training=False)
        
        # Early exit logic
        ee_confidence = tf.reduce_max(tf.nn.softmax(ee_output, axis=-1), axis=-1)
        early_exit_mask = ee_confidence >= self.ee_threshold
        self.early_exit_rate.update_state(tf.cast(early_exit_mask, tf.float32))
        
        # Use early exit predictions where confidence is high
        final_preds = tf.where(
            early_exit_mask[:, tf.newaxis],
            ee_output,
            main_output
        )
        
        # Update metrics
        self.ee_accuracy.update_state(y, ee_output)
        self.main_accuracy.update_state(y, main_output)
        
        return {
            'ee_accuracy': self.ee_accuracy.result(),
            'main_accuracy': self.main_accuracy.result(),
            'early_exit_rate': self.early_exit_rate.result()}
    
    def reset_metrics(self):
        self.ee_accuracy.reset_states()
        self.main_accuracy.reset_states()
        self.early_exit_rate.reset_states()
        
    def _bn_relu(x, name_prefix):
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu")
        return x

    def build_model(self):
        # 1. Initial layers
        self.initial_conv = self.regularized_padded_conv(
            name='initial_conv',
            **self.first_conv
        )
        if self.block_type == 'original':
            self.initial_bn = tf.keras.layers.BatchNormalization(name='initial_bn')
            self.initial_relu = tf.keras.layers.ReLU(name='initial_relu')
        
        # 2. Create blocks up to early exit point
        self.pre_exit_blocks = []
        ee_block_idx, ee_inner_idx = self.ee_location
        
        for block_idx, (group_size, feature, stride) in enumerate(
            zip(self.group_sizes[:ee_block_idx+1], 
                self.features[:ee_block_idx+1], 
                self.strides[:ee_block_idx+1])
        ):
            block_group = []
            for i in range(group_size if block_idx < ee_block_idx else ee_inner_idx + 1):
                prefix = f'pre_block_{block_idx}_{i}'
                block_group.append(self._create_block(feature, stride if i == 0 else 1, prefix))
                if (block_idx, i) == self.ee_location:
                    self.join_point = block_group[-1]['relu']
                    break
            self.pre_exit_blocks.append(block_group)
    
        # 3. Early exit branch
        prefix = f'early_exit_{ee_block_idx}_{ee_inner_idx}'
        self.ee_conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', name=f'{prefix}_conv1')
        self.ee_bn = tf.keras.layers.BatchNormalization(name=f'{prefix}_bn1')
        self.ee_relu = tf.keras.layers.ReLU(name=f'{prefix}_relu1')
        self.ee_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{prefix}_pool1')
        self.ee_flatten = tf.keras.layers.Flatten(name=f'{prefix}_flatten')
        self.ee_dense = tf.keras.layers.Dense(
            self.num_classes, 
            activation="softmax",
            name=f'{prefix}_output'
        )
    
        # 4. Post early exit blocks
        self.post_exit_blocks = []
        for block_idx, (group_size, feature, stride) in enumerate(
            zip(self.group_sizes[ee_block_idx:], 
                self.features[ee_block_idx:], 
                self.strides[ee_block_idx:]),
            start=ee_block_idx
        ):
            block_group = []
            start_idx = ee_inner_idx + 1 if block_idx == ee_block_idx else 0
            for i in range(start_idx, group_size):
                prefix = f'post_block_{block_idx}_{i}'
                block_group.append(self._create_block(feature, stride if i == 0 else 1, prefix))
            if block_group:  # Only append if there are blocks in the group
                self.post_exit_blocks.append(block_group)
    
        # 5. Final layers
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='GMP_layer')
        self.final_dense = tf.keras.layers.Dense(
            self.num_classes,
            kernel_regularizer=self.regularizer,
            name='main_output'
        )
    
    def _create_block(self, feature, stride, prefix):
        """Helper method to create a residual block"""
        return {
            'conv1': self.regularized_padded_conv(
                feature, 3,
                strides=stride,
                name=f'{prefix}_conv1'
            ),
            'bn1': tf.keras.layers.BatchNormalization(
                name=f'{prefix}_bnrelu1_bn'
            ),
            'relu1': tf.keras.layers.ReLU(
                name=f'{prefix}_bnrelu1_relu'
            ),
            'conv2': self.regularized_padded_conv(
                feature, 3,
                name=f'{prefix}_conv2'
            ),
            'bn2': tf.keras.layers.BatchNormalization(
                name=f'{prefix}_bn2'
            ),
            'add': tf.keras.layers.Add(
                name=f'{prefix}_add'
            ),
            'relu': tf.keras.layers.ReLU(
                name=f'{prefix}_relu'
            )
        }

    def regularized_padded_conv(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        return tf.keras.layers.Conv2D(
            *args,
            **kwargs,
            padding='same',
            kernel_regularizer=self.regularizer,
            kernel_initializer='he_normal',
            use_bias=False,
            name=name,
        )

    def shortcut(self, x, filters, stride, name_prefix):
        if x.shape[-1] == filters:
            return x
        elif self.shortcut_type == 'B':
            return self.regularized_padded_conv(filters, 1, strides=stride, name=f"{name_prefix}_conv")(x)
        elif self.shortcut_type == 'B_original':
            x = self.regularized_padded_conv(filters, 1, strides=stride, name=f"{name_prefix}_conv")(x)
            return tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        elif self.shortcut_type == 'A':
            return PadLayer(stride, filters, name=f"{name_prefix}_pad")(x)
        else:
            raise KeyError("Parameter shortcut_type not recognized!")