from __future__ import print_function, division

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
from trajectory_dataset import Loader
import numpy as np
import os



class Pix2Pix():

    def __init__(self):

        # Input shape

        self.file_rows = 256
        self.file_cols = 1
        self.channels = 2 #3 if bearing, 2 if just lat/lon
        self.file_shape = (self.file_rows, self.file_cols, self.channels)

        # Configure data loader

        self.dataset_name = 'trajectories'
        self.data_loader = Loader()
        
        #history to plot
        
        self.history = []
        self.losses = []
        self.acc = []

        # Calculate output shape of D (PatchGAN)

        patch = int(self.file_rows / 2**4)
        self.disc_patch = (patch, 1, 1)


        # Number of filters in the first layer of G and D

        self.gf = 8
        self.df = 8
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator

        self.generator = self.build_generator()

        # Input images and their conditioning images

        file_A = Input(shape=self.file_shape)
        file_B = Input(shape=self.file_shape)

        # By conditioning on B generate a fake version of A

        fake_A = self.generator(file_B)



        # For the combined model we will only train the generator

        self.discriminator.trainable = False



        # Discriminators determines validity of translated images / condition pairs

        valid = self.discriminator([fake_A, file_B])



        self.combined = Model(inputs=[file_A, file_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)


    def build_generator(self):

        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):

            """Layers used during downsampling"""

            d = Conv2D(filters, kernel_size=(f_size,1), strides=(2,1), padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            #print(d.shape)
            return d



        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):

            """Layers used during upsampling"""

            u = UpSampling2D(size=(2,1))(layer_input)
            u = Conv2D(filters, kernel_size=(f_size,1), strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            #print(u.shape)
            return u


        d0 = Input(shape=self.file_shape)

        # Downsampling

        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling

        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=(2,1))(u6)
        
        #print(u7.shape)

        output_file = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

        return Model(d0, output_file)



    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):

            """Discriminator layer"""

            d = Conv2D(filters, kernel_size=(f_size,1), strides=(2,1), padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        file_A = Input(shape=self.file_shape)
        file_B = Input(shape=self.file_shape)

        # Concatenate file and conditioning file by channels to produce input

        combined_files = Concatenate(axis=-1)([file_A, file_B])

        d1 = d_layer(combined_files, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=(4,1), strides=1, padding='same')(d4)

        return Model([file_A, file_B], validity)



    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            
            for batch_i, (files_A, files_B) in enumerate(self.data_loader.load_batch(batch_size)):
                """
                files_A = full trajectory
                files_B = corrupt trajectory, a condition
                """
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                #print(np.array(files_B).shape, np.array(files_B).shape)
                
                fake_A = self.generator.predict(files_B)

                # Train the discriminators (original images = real / generated = Fake)

                d_loss_real = self.discriminator.train_on_batch([files_A, files_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, files_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------
                
                g_loss = self.combined.train_on_batch([files_A, files_B], [valid, files_A])
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress

                print ("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0], elapsed_time))

                # If at save interval => save generated image samples

                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    gl = g_loss[0]
                    if gl>300:
                        gl=300
                    self.losses.append({"D":d_loss[0],"G":g_loss[0]})
                    self.acc.append({"Accuracy":d_loss[1]*100})



    def sample_images(self, epoch, batch_i):

        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        avg = 0
        for batch_i, (files_A, files_B) in enumerate(self.data_loader.load_batch(3)):
            
            fake_A = self.generator.predict(files_B)
            
            files_A = files_A.reshape(self.file_rows,self.channels)
            fake_A = fake_A.reshape(self.file_rows,self.channels)
            files_B = files_B.reshape(self.file_rows,self.channels)
            avg += self.data_loader.save_generated_data(epoch, batch_i, files_B, files_A, fake_A)
        self.history.append(({"Average distance":avg/3}))
        
    def plot_loss(self):
        
        #plot average distance between points of gen/real file, losses in each epoch, accuracy
        
        self.data_loader.plot_losses(self.history)
        self.data_loader.plot_losses(self.losses)
        self.data_loader.plot_losses(self.acc)




if __name__ == '__main__':

    gan = Pix2Pix()

    gan.train(epochs=1000, batch_size=1, sample_interval=200)
    
    gan.plot_loss()