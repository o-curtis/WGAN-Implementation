# WGAN-Implementation
Implementation of WGAN. Originally used for (Curtis et. al, 2021)

class astrofake():
    def __init__(self, data, image_shape=(256,256,1), kernel_size=(4,4), batch_size=16, latent_dim=200, starting_num_filters=int(512), every_n_epochs=5):
        """
        Initialize WGAN.
        Parameters
        data - arr - 4d array of shape (N, image_shape). i.e., a 4d tensor of monochannel 2 dimensional images.
        image_shape - touple - touple of shape (dim, dim, 1) where dim is the resolution of your images.
        kernel_size - touple - touple of shape (k, k) where k is the size of the kernel used for convolution layers
        batch_size - int - How many samples the network is trained on before updating the parameters
        latent_dim - int - latent_dim x latent_dim is the size of the latent space vector fed into the generator network
        starting_num_filters - int - how many filters the generator network starts with / the discriminator network ends with
        every_n_epochs - int - How often to save the loss of the networks
        """
    T
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
