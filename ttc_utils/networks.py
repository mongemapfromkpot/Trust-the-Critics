"""
Zoo of discriminator models for use with TTC
"""
from torch import nn


##################
# Simple convolutional block, used by several networks
##################
class ConvBlock(nn.Module):
    def __init__(self, chan_in = 3, chan_out = 32, ker_size = 3, stride = 1, pad = 1):
        super(ConvBlock, self).__init__()

        self.main = nn.Sequential(nn.Conv2d(chan_in, chan_out, ker_size, stride, padding = pad),
                            nn.LeakyReLU(negative_slope = 0.1))
    def forward(self, input):
        return self.main(input)
      
      
##################
# Pytorch version of "ConvNetClassifier" from Lunz et. al. Adversarial Regularizers in Inverse Problems
##################
class arConvNet(nn.Module):
    def __init__(self, DIM, shape):
        super(arConvNet, self).__init__()
        
        num_chan, h, w = shape
        self.h = h
        self.w = w

        conv1 =  ConvBlock(chan_in = num_chan, chan_out =16, ker_size = 5, stride = 1, pad = 2)#produces 16xhxw
        conv2 =  ConvBlock(chan_in = 16, chan_out =32, ker_size = 5, stride = 1, pad = 2)#produces 32xhxw
        conv3 =  ConvBlock(chan_in = 32, chan_out =32, ker_size = 5, stride = 2, pad = 2)#produces 32xh/2xw/2
        conv4 =  ConvBlock(chan_in = 32, chan_out =64, ker_size = 5, stride = 2, pad = 2)#produces 64xh/4xw/4
        conv5 =  ConvBlock(chan_in = 64, chan_out =64, ker_size = 5, stride = 2, pad = 2)#produces 64xh/8xw/8
        conv6 =  ConvBlock(chan_in = 64, chan_out =128, ker_size = 5, stride = 2, pad = 2)#produces 128xh/16xw/16
        
        self.main = nn.Sequential(conv1,conv2,conv3,conv4,conv5,conv6)
                    
        self.linear1 = nn.Sequential(nn.Linear(128*(h//16)*(w//16), 256), nn.LeakyReLU(negative_slope = 0.1))
        self.linear2 = nn.Linear(256,1)

    def forward(self, input):

        output = self.main(input)
        output = output.view(-1, 128*(self.h//16)*(self.w//16))
        output = self.linear1(output)
        output = self.linear2(output)
        return output


################
# SNDC discriminator, from "A Large Scale Study of Regularization and Normalization in GANs"
################
class sndcgan(nn.Module):
    def __init__(self, DIM, shape):
        super(sndcgan, self).__init__()
        
        num_chan, h, w = shape
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        self.main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 1 * DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 2 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 4 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 4 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 1, pad = 1))
        self.linear = nn.Linear(self.final_h*self.final_w*8*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*8*self.DIM)
        output = self.linear(output)
        return output   
    
    
################
# BSNDC discriminator, which is SNDC discriminator with more convolutions
################
class bsndcgan(nn.Module):
    def __init__(self, DIM, shape):
        super(bsndcgan, self).__init__()
        
        num_chan, h, w = shape
        self.DIM = DIM
        self.final_h = max(h//128,1)
        self.final_w = max(w//128,1)
        main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 1 * DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 2 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 2 * DIM, chan_out = 4 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 4 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 4 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 1, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 3, stride = 2, pad = 1),
                ConvBlock(chan_in = 8 * DIM, chan_out = 8 * DIM, ker_size = 4, stride = 1, pad = 0),
                )
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*8*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*8*self.DIM)
        output = self.linear(output)
        return output


###############
# DCGAN discriminator, from Radford et. al. 
###############
class dcgan(nn.Module):
    def __init__(self, DIM, shape):
        super(dcgan, self).__init__()
       
        num_chan, h, w = shape
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        main = nn.Sequential(nn.Conv2d(num_chan, DIM, 3, 2,padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(DIM, 2*DIM, 3, 2, padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding = 1),
                nn.LeakyReLU())
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*4*self.DIM)
        output = self.linear(output)
        return output

# We also include the InfoGAN generator for WGAN experiments
class dcgan_generator(nn.Module):#added bias = False to each module followed by a batchnorm, as batchnorm will send mean to 0, effectively cancelling any learned bias
    def __init__(self, DIM, num_chan, h, w):
        super(dcgan_generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM, bias = False),#latent dimension is 128
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True)
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2, bias = False),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2, bias = False),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, num_chan, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()
        self.DIM = DIM
        self.num_chan = num_chan
        self.latent_dim = 128

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.num_chan, 32, 32)


################
# InfoGAN discriminator, from "Are GANs created equal?"
################
class infogan(nn.Module):
    def __init__(self, DIM, shape):
        super(infogan, self).__init__()
        
        num_chan, h, w = shape
        self.DIM = DIM
        self.final_h = h//4
        self.final_w = w//4
        main = nn.Sequential(ConvBlock(chan_in = num_chan, chan_out = DIM, ker_size = 4, stride = 2, pad = 1),
                ConvBlock(chan_in = DIM, chan_out = 2 * DIM, ker_size = 4, stride = 2, pad = 1),
                )
        self.main = main
        self.fc1 = nn.Sequential(nn.Linear(self.final_h*self.final_w*2*DIM, 1024),
                nn.LeakyReLU(negative_slope = 0.1))
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*2*self.DIM)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

# We also include the InfoGAN generator for WGAN experiments
class infogan_generator(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(infogan_generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(64, 1024, bias = False),#latent dimension is 64
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024,128 * (h // 4) * (w // 4) , bias = False),
            nn.BatchNorm1d(128 * (h // 4) * (w // 4)),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, num_chan, 4, stride=2, bias = False, padding = 1),
        )

        self.tanh = nn.Tanh()
        self.num_chan = num_chan
        self.h = h
        self.w = w
        self.latent_dim = 64

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        output = output.view(-1, 128, self.h // 4, self.w // 4)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.tanh(output)
        return output.view(-1, self.num_chan, self.h, self.w)