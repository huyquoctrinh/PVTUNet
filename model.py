import tensorflow as tf
from keras_cv_attention_models import caformer
from layers.upsampling import decode
from layers.convformer import convformer
from layers.util_layers import merge, conv_bn_act, bn_act
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K
from keras_cv_attention_models import convnext, efficientnet, pvt
from keras_cv_attention_models import attention_layers

def decode1(inputs, filters, activation='swish', padding='same'):
    
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    
    return x

def build_model(img_size = 352, num_classes = 1):
    
    backbone = pvt.PVT_V2B0(input_shape=(img_size, img_size, 3), pretrained="imagenet", num_classes = 0)
    layer_names = ['stack4_block2_output_ln', 'stack3_block2_output_ln', 'stack2_block2_output_ln', 'stack1_block1_attn_ln']

    layers = [backbone.get_layer(x).output for x in layer_names]

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    x = layers[0]

    for i, layer in enumerate(layers[1:]):

        x = decode1(x ,layer.shape[channel_axis])

        x = merge([x, layer], layer.shape[channel_axis])
        x = conv_bn_act(x, layer.shape[channel_axis], (3, 3))

    filters = layers[-1].shape[channel_axis] 

    x = decode1(x, x.shape[channel_axis])
    x = conv_bn_act(x, layer.shape[channel_axis], (3, 3))
    x = decode1(x ,x.shape[channel_axis])
    x = conv_bn_act(x, layer.shape[channel_axis], (3, 3))

    x = Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid', name = "mask_out_4")(x)


    model = Model(inputs = backbone.input, outputs = x)
    return model