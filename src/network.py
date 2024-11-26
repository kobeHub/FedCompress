def get_resnet20_network(input_shape=(32,32,3), num_classes=10, weights_dir=None, ee_threshold=1.0,
						  ee_location=None, new_imple=False):
	if ee_location is None:
		from architectures import Resnets
		return Resnets.resnet20(input_shape=input_shape, num_classes=num_classes, 
							 weights_dir=weights_dir)
	else:
		from architectures.ResNetsEEV2 import ResNetEE
		return ResNetEE.resnet20_ee(input_shape=input_shape, num_classes=num_classes, weights_dir=weights_dir, 
						ee_location=ee_location, ee_threshold=ee_threshold)

def get_yamnet_network(input_shape=(None,40,1), num_classes=10, weights_dir=None):
	from architectures import YAMNet
	return YAMNet.create_yamnet_model(input_shape=input_shape, num_classes=num_classes, weights_dir=weights_dir)

def get_model(name='resnet20'):
	if name=='resnet20':
		return get_resnet20_network
	elif name=='yamnet':
		return get_yamnet_network
	else:
		raise Exception("Model `{}` is not available. Please provide one of [`resnet20`,`cnn`,`yamnet`].".format(name)) 