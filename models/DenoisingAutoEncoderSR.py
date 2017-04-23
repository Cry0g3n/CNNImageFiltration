from keras.engine import Input, Model
from keras.layers import merge, Conv2D, Deconv2D

from models.BaseSuperResolutionModel import BaseSuperResolutionModel


class DenoisingAutoEncoderSR(BaseSuperResolutionModel):
    def __init__(self):
        super(DenoisingAutoEncoderSR, self).__init__("DenoiseAutoEncoderSR")

        self.n1 = 64
        self.f1 = 3
        self.f2 = 5

    def create_model(self, height=64, width=64, channels=1, batch_size=32):
        self.height = height
        self.width = width
        self.channels = channels

        init = Input(shape=(height, width, channels))

        level1_1 = Conv2D(self.n1, [self.f1, self.f1], activation='relu', padding='same')(init)
        level2_1 = Conv2D(self.n1, [self.f1, self.f1], activation='relu', padding='same')(level1_1)

        level2_2 = Deconv2D(self.n1, [self.f1, self.f1], activation='relu', padding='same')(
            level2_1)
        level2 = merge([level2_1, level2_2], mode='sum')

        level1_2 = Deconv2D(self.n1, [self.f1, self.f1], activation='relu', padding='same')(level2)
        level1 = merge([level1_1, level1_2], mode='sum')

        decoded = Conv2D(channels, [self.f2, self.f2], activation='linear', padding='same')(level1)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

        return model

    def fit(self, batch_size=32, nb_epochs=1):
        return super(DenoisingAutoEncoderSR, self).fit(batch_size, nb_epochs)
