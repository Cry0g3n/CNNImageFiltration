from keras.engine import Input, Model
from keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Activation

from models.BaseSuperResolutionModel import BaseSuperResolutionModel


class DnCNN(BaseSuperResolutionModel):
    def __init__(self):
        super(DnCNN, self).__init__("DnCNN")

        self.n1 = 64
        self.f1 = 3
        self.layers = 15

    def create_model(self, height=64, width=64, channels=1, batch_size=32):
        self.height = height
        self.width = width
        self.channels = channels

        init = Input(shape=(height, width, channels))

        level1_1 = Conv2D(self.n1, [self.f1, self.f1], activation='relu', padding='same', use_bias=False)(init)

        level2_1 = SeparableConv2D(self.n1, [self.f1, self.f1], padding='same')(level1_1)
        level2_2 = BatchNormalization()(level2_1)
        level2_3 = Activation('relu')(level2_2)

        decoded = Conv2D(channels, [self.f1, self.f1], activation='linear', padding='same')(level2_3)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

        return model

    def fit(self, batch_size=32, nb_epochs=1):
        return super(DnCNN, self).fit(batch_size, nb_epochs)

    def predict(self, image, stride=32):
        noise = super(DnCNN, self).predict(image, stride)
        f_image = image - noise

        return f_image

    def save_data(self, x_name, y_name):
        train_data = self.raw_data

        x_train = []
        y_train = []

        for data_patch in train_data:
            x_train.append(data_patch[x_name])
            y_train.append(data_patch[x_name] - data_patch[y_name])

        self.x_data = self.prepare_data(x_train)
        self.y_data = self.prepare_data(y_train)