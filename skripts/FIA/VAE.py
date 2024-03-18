import os
import keras
from keras import Model
from keras.layers import Input, Dense, LeakyReLU, Lambda, Dropout, BatchNormalization
from keras.losses import mse, kullback_leibler_divergence
from keras.optimizers import Nadam
import keras.backend as backend

class mtVAE():

    def __init__(self, input_shape, intermediate_neurons:int, latent_dim:int,       # Shape
                kl_beta:float=1e-2, learning_rate:float=1e-3, intermediate_dropout:float=0.5, input_dropout:float=0.5,
                intermediate_activation=LeakyReLU()):                     # Rates
        
        self.input_shape = input_shape
        self.intermediate_neurons = intermediate_neurons
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta
        self.learning_rate = learning_rate
        self.input_dropout = input_dropout
        self.intermediate_dropout = intermediate_dropout
        self.intermediate_activation = intermediate_activation
        
        # # =================
        # # Encoder
        # # =================
 
        self.input      = Input(shape=(self.input_shape,), name='encoder_input')
        self.input      = Dropout(self.intermediate_dropout) (self.input)
        self.enc        = Dense(self.intermediate_neurons, activation=self.intermediate_activation) (self.input)
        self.enc        = Dropout(self.intermediate_dropout) (self.enc)                                                 # Dropout for more redundant neurons
        self.enc        = BatchNormalization() (self.enc)                                            
        self.mu         = Dense(self.latent_dim, name='latent_mu') (self.enc)
        self.sigma      = Dense(self.latent_dim, name='latent_sigma') (self.enc)

        self.z          = Lambda(self.sample_z, output_shape=(self.latent_dim, ), name='z')([self.mu, self.sigma])      # Use reparameterization trick 

        self.encoder = Model(self.input, [self.mu, self.sigma, self.z], name='encoder')                                 # Instantiate encoder

        # =================
        # Decoder
        # =================

        # Definition
        self.decoder_input  = Input(shape=(self.latent_dim, ), name='decoder_input')
        self.dec            = Dense(self.intermediate_neurons, activation=self.intermediate_activation) (self.decoder_input) 
        self.dec            = Dropout(self.intermediate_dropout) (self.dec)
        self.dec            = BatchNormalization() (self.dec)
        self.output  = Dense(self.input_shape) (self.dec )

        self.decoder = Model(self.decoder_input, self.output, name='decoder')                                           # Instantiate decoder

        # =================
        # VAE
        # =================

        # Instantiate VAE
        self.vae_outputs = self.decoder(self.encoder(self.input)[2])
        self.vae         = Model(self.input, self.vae_outputs, name='vae')

        # Define optimizer
        self.optimizer = Nadam(learning_rate=self.learning_rate)

        # Compile VAE
        self.vae.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss, metrics = ['mse'])
    
    
    def train(self, train_data, val_data, n_epochs, batch_size, verbosity=1):
        self.vae.fit(train_data, train_data,
                     epochs = n_epochs, 
                     batch_size = batch_size, 
                     validation_data = (val_data, val_data),
                     verbose = verbosity)
    
    def encode(self, data):
        return self.encoder.predict(data)[2]
    
    def encode_mu(self, data):
        return self.encoder.predict(data)[0]
    
    def decode(self, data):
        return self.decoder.predict(data)
    
    def reconstruct(self, data):
        return self.decode(self.encode(data))
    
    def save_model(self, save_folder, suffix:str=""):
        self.vae.save(os.path.join(save_folder, f'VAE{suffix}.h5'))
        self.encoder.save(os.path.join(save_folder, f'VAE_encoder{suffix}.h5'))
        self.decoder.save(os.path.join(save_folder, f'VAE_decoder{suffix}.h5'))
        
    
    def load_vae(self, save_path):
        # The two functions below have to be redefined for the loading
        # of the model. They cannot be methods of the mtVAE class for
        # some reason.
        # https://github.com/keras-team/keras/issues/13992
                       
        self.vae = keras.models.load_model(save_path)
        self.vae.compile(optimizer=self.optimizer, 
                         custom_objects={'sample_z': self.sample_z}, 
                         loss=self.kl_reconstruction_loss, 
                         metrics = ['mse'])
        
    def load_encoder(self, save_path):
        self.encoder = keras.models.load_model(save_path)
        
    def load_decoder(self, save_path):
        self.decoder = keras.models.load_model(save_path)

    def sample_z(self, args):
        """
        Define sampling with reparameterization trick
        """
        mu, sigma = args
        batch     = backend.shape(mu)[0]
        dim       = backend.int_shape(mu)[1]
        eps       = backend.random_normal(shape=(batch, dim))
        return mu + backend.exp(sigma / 2) * eps
    
    def kl_reconstruction_loss(self, true, pred):
        """
        Kullback-Leibler + Reconstruction loss
        """
        # Reconstruction loss
        reconstruction_loss = mse(true, pred) * self.input_shape

        # KL divergence loss
        kl_loss = 1 + self.sigma - backend.square(self.mu) - backend.exp(self.sigma)
        kl_loss = backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Total loss = mean(rec + scaler * KL divergence loss )
        return backend.mean(reconstruction_loss + self.kl_beta * kl_loss)