from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, SeparableConv2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import SpatialDropout2D


def EEGNetBranch(input_shape, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    eeg_input = Input(shape=input_shape, name='eeg_input')

    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(eeg_input)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                              depth_multiplier=D,
                              depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    return eeg_input, flatten


def fNIRSBranch(input_shape, dropoutRate=0.5):
    fnirs_input = Input(shape=input_shape, name='fnirs_input')

    x = Dense(64, activation='relu')(fnirs_input)
    x = Dropout(dropoutRate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Flatten()(x)
    return fnirs_input, x


def MultimodalEEGNetWorkload(eeg_shape=(32, 500, 1), fnirs_shape=(36, 11), nb_classes=3):
    eeg_input, eeg_features = EEGNetBranch(eeg_shape)
    fnirs_input, fnirs_features = fNIRSBranch(fnirs_shape)

    merged = Concatenate()([eeg_features, fnirs_features])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(nb_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[eeg_input, fnirs_input], outputs=output)
    return model


if __name__ == '__main__':
    model = MultimodalEEGNetWorkload()
    model.summary()
