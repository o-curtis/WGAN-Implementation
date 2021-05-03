import numpy as np
from matplotlib import pyplot as plt
import h5py
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, UpSampling2D, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, LeakyReLU, ReLU, GaussianNoise, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.backend import mean as TFmean
from glob import glob
from scipy.ndimage import gaussian_filter

from clip_constraint import *

class astrofake():  

    def __init__(self, data, image_shape=(256,256,1), kernel_size=(4,4), batch_size=16, latent_dim=200, starting_num_filters=int(512), every_n_epochs=5):
        """
        Initialize GAN.
        Attempts to load in data from .npy file. If it cannot then it will preprocess Gadget-2 snapshot files
        Saves the loaded density maps in self.real_data.

        Parameters
        ----------
        data - arr - 4d array of shape (N, image_shape). i.e., a 4d tensor of monochannel 2 dimensional images.
        image_shape - touple - touple of shape (dim, dim, 1) where dim is the resolution of your images.
        kernel_size - touple - touple of shape (k, k) where k is the size of the kernel used for convolution layers
        batch_size - int - How many samples the network is trained on before updating the parameters
        latent_dim - int - latent_dim x latent_dim is the size of the latent space vector fed into the generator network
        starting_num_filters - int - how many filters the generator network starts with / the discriminator network ends with
        every_n_epochs - int - How often to save the loss of the networks
        """

        self.params = {
            'latent_dim': latent_dim,
            'start_num_filters': starting_num_filters,
            'batch_size': batch_size,
            'every_n_epochs': every_n_epochs
        }


        self.img_shape = image_shape
        self.kernel_size = kernel_size
        self.real_data = data
       
        np.random.shuffle(self.real_data)
        self.sample_size = len(self.real_data)
        
        #define the optimizer
        d_optimizer = Adam(learning_rate=1e-6, beta_1=0.5)
        g_optimizer = Adam(learning_rate=5e-5, beta_1=0.5)
        #d_optimizer  = RMSprop(lr=1e-6)
        #g_optimizer  = RMSprop(lr=5e-5)
        
        #build and compile discriminator
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss=wasserstein_loss, optimizer=d_optimizer, metrics=['accuracy'])
        
        #build the generator
        self.generator = self.create_generator()
        
        #build the combined GAN
        z = Input(shape=(self.params['latent_dim'],))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss=wasserstein_loss, optimizer=g_optimizer)
        
        #Try to load previously trained networks
        try:
            gen_models = glob("gen*.h5")
            print("GEN MODELS: ", gen_models)
            self.generator = load_model(gen_models[-1])
            print("loaded previously trained generator model")
        except:
            pass
        try:
            dis_models = glob("dis*.h5")
            self.discriminator = load_model(dis_models[-1])
            print("loaded previously trained discriminator model")
        except:
            pass
        try:
            com_models = glob("com*.h5")
            self.combined = load_model(com_models[-1])
            print("loaded previously trained combined model")
        except:
            pass
        
    def create_generator(self):
        """
        Initializes the generator network consisting of 5 hidden layers. 
        The input layer is a latent_dim x start_num_filters (default 200x512) gaussian distributed tensor.
        The first hidden layer is a dense layer that reshapes the input into a 16x16x512 low res density map
        The second hidden layer is a conv2DTranspose layer that upscales the input to a 32x32x256 tensor
        The third hidden layer is a conv2DTranspose layer that upscales the image to a 64x64x128 tensor
        The fourth hidden is a conv2DTranspose layer that upscales the image to a 128x128x64 tensor
        The fifth hidden layer is a conv2DTranspose layer that upscales the image to a 256x256x1 tensor
        
        Hidden layers 2-4 are batch normalized with a ReLU activation function.
        The fifth hidden layer uses a tanh activation function. TODO: change this to our custom normalization function.
        
        """
        generator=Sequential()
        
        n_nodes = self.params['start_num_filters'] * 16 * 16 #todo, don't hard code this
        
        generator.add(Dense(n_nodes, input_dim=self.params['latent_dim'], activation='relu'))
        generator.add(Reshape((16,16,self.params['start_num_filters'])))
 
        generator.add(Conv2DTranspose(int(self.params['start_num_filters']/2.), self.kernel_size, strides=(2,2), padding='same', activation='relu'))
        generator.add(BatchNormalization(momentum=0.8))

        generator.add(Conv2DTranspose(int(self.params['start_num_filters']/4.), self.kernel_size, strides=(2,2), padding='same', activation='relu'))
        generator.add(BatchNormalization(momentum=0.8))

        generator.add(Conv2DTranspose(int(self.params['start_num_filters']/8.), self.kernel_size, strides=(2,2), padding='same', activation='relu'))
        generator.add(BatchNormalization(momentum=0.8))
        
        generator.add(Conv2DTranspose(1, self.kernel_size, strides=(2,2), activation='tanh', padding='same'))
        
        generator.summary()
        
        noise = Input(shape=(self.params['latent_dim'],))
        img = generator(noise)
        
        return Model(noise, img)

    def create_discriminator(self):
        """
        Initializes the discriminator network consisting of 5 hidden layers.
        The input layer is a dim x dim x 1 tensor (default 256 x 256 x 1) (last dimension is filters).
        The first hidden layer is a conv layer that downscales the image to a 128x128x64 tensor.
        The second hidden layer is a conv layer that downscales the image to a 64x64x128 tensor.
        The third hidden layer is a conv layer that downscales the image to a 32x32x256 tensor
        The fourth hidden layer is a conv layer that downscales the image to a 16x16x256 tensor
        The final layer is a dense layer with 1 node.
        
        Hidden layers 1-4 are batch normalized with a LeakyReLU activation function.
        Hidden layer 5 has a sigmoid activation function.
        
        The loss is a binary-crossentropy loss optimized with an Adam optimizer.
        """
        const = ClipConstraint(0.01)
        
        discriminator=Sequential()
        
        discriminator.add(Conv2D(int(self.params['start_num_filters']/8.),  self.kernel_size, kernel_constraint=const, \
                                 input_shape=self.img_shape, strides=(2,2), padding='same'))
        discriminator.add(LeakyReLU(0.2))
        
        discriminator.add(Conv2D(int(self.params['start_num_filters']/4.), self.kernel_size, strides=(2,2), \
                                 kernel_constraint=const, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))
        
        discriminator.add(Conv2D(int(self.params['start_num_filters']/2.), self.kernel_size, strides=(2,2), \
                                 kernel_constraint=const, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))
        
        discriminator.add(Conv2D(int(self.params['start_num_filters']), self.kernel_size, strides=(2,2), \
                                 kernel_constraint=const, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Flatten())        
        discriminator.add(Dense(units=1, activation='linear'))       

        discriminator.summary()
        
        img = Input(shape=self.img_shape)
        validity = discriminator(img)
        
        return Model(img, validity)
    
    def train_a_network(self, epochs=[0,20], snapshot_number=10):
        """
        Main method to train the GAN. For each epoch we iterate over the training set by N=batch_size/2 samples
        at a time. The real_data is randomly shuffled after each epoch. We then train the discriminator
        network on a set of fake data and then on a set of real data using batch normalization
        Finally, the generator network is trained using batch normalization. When training is complete 
        the generator network and discriminator network parameters are saved as .h5 files.
        
        Parameters
        ----------
        epochs - int - total number of epochs to train over
        snapshot_number - int - which snapshot number (i.e., which time step in the sim) are we training the GAN on.
            used for outputting network files.
       
        Returns
        ----------
        g_loss - array - loss of generator network training.
            Size = num_snaps * dim / batch_size rounded up 
        
        d_loss_reals - array - loss of discriminator network training while learning from real images.
            Size = num_snaps * dim / batch_size rounded up 
        
        d_loss_fakes - array - loss of discriminator network training while training from fake images.
            Size = num_snaps * dim / batch_size rounded up 

        """   
        d_loss = []
        g_loss = []
        half_batch = int(self.params['batch_size']/2)
        for e in range(epochs[0],epochs[1]):
            used_this_batch = np.arange(self.sample_size)
            np.random.shuffle(used_this_batch)
            for this_batch in range(int(self.sample_size / half_batch)):
                tmp_d = []
                for _ in range(5):
                    #get a random half batch of real samples
                    real_samples = self.real_data[used_this_batch[this_batch * half_batch : (this_batch+1) * half_batch]] 
                    real_labels  = -np.ones((len(real_samples),1))

                    #generate fake samples
                    noise = np.random.normal(0, 1, (half_batch, self.params['latent_dim']))
                    gen_imgs = self.generator.predict(noise)
                    fake_labels = np.ones((len(noise),1))

                    #train the discriminator on half batch of real and fake images, add loss to list
                    this_real_loss = self.discriminator.train_on_batch(real_samples, real_labels)
                    this_fake_loss = self.discriminator.train_on_batch(gen_imgs, fake_labels)
                    this_d_loss = 0.5*np.add(this_real_loss, this_fake_loss)
                    tmp_d.append(this_d_loss)
                d_loss.append(np.mean(tmp_d,axis=0))    
                
                #train the generator on a full batch of noise
                noise = np.random.normal(0, 1, (self.params['batch_size'], self.params['latent_dim']))
                real_labels = -np.ones((self.params['batch_size'],1))
                this_g_loss = self.combined.train_on_batch(noise, real_labels)
                g_loss.append(this_g_loss)
                
                #print("Epoch: %d Iteration: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (e, this_batch, this_d_loss[0], 100*this_d_loss[1], this_g_loss))
                   
            if (e==0) or (e%self.params['every_n_epochs']==0):
                noise = np.random.normal(0, 1, (self.params['batch_size'], self.params['latent_dim']))
                test_image = self.generator.predict(noise)
                np.save("test_"+str(e), test_image)
                plt.clf()
                if self.params['interactive_plots']:
                    plt.imshow(test_image[0,:,:,0])
                    plt.show()
                    plt.clf()
                #self.generator.save("generator."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
                #self.discriminator.save("discriminator."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
                #self.combined.save("combined."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
                #self.generator = load_model("generator."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
                #self.discriminator = load_model("discriminator."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
                #self.combined = load_model("combined."+str(snapshot_number).zfill(3)+"."+str(e).zfill(3)+".h5")
            np.savetxt('gloss.dat', g_loss)
            np.savetxt('dloss.dat', d_loss)
        self.generator.save("generator."+str(snapshot_number).zfill(3)+".h5")
        self.discriminator.save("discriminator."+str(snapshot_number).zfill(3)+'.h5')
        self.combined.save('combined.'+str(snapshot_number).zfill(3)+'.h5')
        return g_loss, d_loss
        
    def load_in_all_density_maps(self):
        """
        This method will load in all Gadget-2 Snapshot files, saving the particle positions as Numpy arrays.
        
        Parameters
        -----------
        
        Returns
        -----------
        density_maps - array - An array of shape (number_of_snapshots, 512^3, 512^3, 512^3) 
            containing the x, y, z coordinates of every particle in all Gadget-2 Snapshot files.
        """  
        snapshot_files = glob(self.params['filename_base'])
        density_maps = []
        for k in range(len(snapshot_files)):
            this_snap = Gadget2SnapshotDE.open(snapshot_files[k])
            this_snap = this_snap.getPositions().value
            density_maps.append(this_snap)
        density_maps = np.array(density_maps)
        return density_maps

def this_normalization(thismap, a=4):
    return ((2*thismap)/(thismap+a)) - 1
    
def inverse_normalization(thismap, a):
    return - ((thismap+1)*a)/(thismap-1)

def wasserstein_loss(y_true, y_pred):
    return TFmean(y_true * y_pred)

if __name__ == '__main__': 
    data = np.load("/projectnb/ganvoid/runs/20200713/all_2d_slices_512p_512mpc_cic_smoothed_normed.npy")
    astrogan = astrofake(batch_size=16, data=data, latent_dim=100, interactive_plots=False, every_n_epochs=1)
    gloss, dloss = astrogan.train_a_network(epochs=[0,4])

