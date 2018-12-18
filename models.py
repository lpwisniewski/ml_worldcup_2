import tensorflow as tf
from tensorflow.python.keras import models


def ternaus_model_building(img_shape):
    inputs = layers.Input(shape=img_shape)
    model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for key in layer_dict:
        print(key)
        print(layer_dict[key])
        print(layer_dict[key].output_shape)
        layer_dict[key].trainble = False

    num_filters = 32

    def decoder_block(input_tensor, concat_tensor, num_filters_a, num_filters_b, up_scale=2):
        decoder = layers.Conv2DTranspose(num_filters_a, (3, 3),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Conv2D(num_filters_b, (3, 3), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        return decoder

    actual_inputs = layers.Conv2D(512, (3, 3), padding='same')(layer_dict['block5_pool'].output)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block5_conv3'].output, 256, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block4_conv3'].output, 256, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block3_conv3'].output, 128, 256)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block2_conv2'].output, 64, 128)
    final = decoder_block(actual_inputs, layer_dict['block1_conv2'].output, 32, 1)

    model = models.Model(inputs=[inputs], outputs=[final])
    return model


def resnet_model_building(img_shape):
    inputs = layers.Input(shape=img_shape)
    model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for key in layer_dict:
        print(key)
        print(layer_dict[key])
        print(layer_dict[key].output_shape)
        layer_dict[key].trainble = False

    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )

    num_filters = 32

    def decoder_block(input_tensor, concat_tensor, num_filters, up_scale=2, cropping=None):
        decoder = layers.Conv2DTranspose(num_filters, (up_scale, up_scale),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        if cropping is not None:
            decoder = layers.Cropping2D(((cropping, 0), (cropping, 0)))(decoder)
        decoder = layers.concatenate([decoder, concat_tensor], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        return decoder

    actual_inputs = layers.MaxPooling2D((2, 2), strides=(2, 2))(layer_dict['add_15'].output)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn5c_branch2c'].output, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn4f_branch2c'].output, 256)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn3d_branch2c'].output, 64)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn2c_branch2c'].output, 32)

    actual_inputs = layers.Conv2DTranspose(num_filters * 2, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2 * 2, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(actual_inputs)

    actual_inputs = layers.Conv2DTranspose(num_filters, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters, (3, 3), padding='same')(actual_inputs)

    final = layers.Conv2D(1, (3, 3), padding='same', activation='softmax')(actual_inputs)

    model = models.Model(inputs=[inputs], outputs=[final])
    tf.keras.utils.plot_model(
        model,
        to_file='final_model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )
    return model


def convolutional_model_building(img_shape,
                                 convolution_size,  # TODO Can be parametrized for each layer
                                 activation_layer,
                                 filters_nb_list,
                                 filters_scaling,
                                 filters_nb_center):
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            input_tensor)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation(activation_layer)(encoder)
        print(encoder)
        encoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            encoder)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation(activation_layer)(encoder)
        print(encoder)
        return encoder

    def encoder_block(input_tensor, num_filters, down_scale=2):
        encoder = conv_block(input_tensor, num_filters)
        print(encoder)
        encoder_pool = layers.MaxPooling2D((down_scale, down_scale),
                                           strides=(down_scale, down_scale))(encoder)
        print(encoder_pool)

        return encoder_pool, encoder

    def decoder_block(input_tensor, concat_tensor, num_filters, up_scale=2):
        decoder = layers.Conv2DTranspose(num_filters, (up_scale, up_scale),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation_layer)(decoder)
        decoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation_layer)(decoder)
        decoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation_layer)(decoder)
        return decoder

    inputs = layers.Input(shape=img_shape)

    encoders = [None] * len(filters_nb_list)

    actual_inputs = inputs
    for i in range(0, len(filters_nb_list)):
        a, b = encoder_block(actual_inputs, filters_nb_list[i], filters_scaling[i])
        encoders[i] = b
        actual_inputs = a

    center = conv_block(actual_inputs, filters_nb_center)

    actual_inputs = center
    for i in reversed(range(0, len(filters_nb_list))):
        print(encoders)
        actual_inputs = decoder_block(actual_inputs,
                                      encoders[i],
                                      filters_nb_list[i],
                                      filters_scaling[i])

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(actual_inputs)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
