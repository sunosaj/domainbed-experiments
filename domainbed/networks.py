# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy
import random


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

        
class VAE(nn.Module):

    def __init__(self, latent_size,
                 encoder_layer_chans=[1, 8, 16, 32], decoder_layer_chans=[32,16,8,1],
                 conditional=False, num_labels=10):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_chans) == list
        assert type(latent_size) == int
        assert type(decoder_layer_chans) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_chans, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_chans, latent_size, conditional, num_labels)

    def forward(self, x=None, d=None, yhat=None, mode='recon'):

        if mode == 'recon':
            means, log_var, d, yhat = self.encoder(x, d)
            z = self.reparameterize(means, log_var)
            recon_x = self.decoder(z, d, yhat)
            return recon_x, means, log_var, d, yhat
        elif mode == 'dom_aug': # unseen domain data generation
            d = torch.zeros([32]).to('cuda')  # for test_env 0, all d = 0
            yhat = torch.zeros([32, 10]).to('cuda')
            y = []
            for i in range(32):
                _y = random.randint(0, 9)
                yhat[i, _y] = 1
                y.append(_y)
            z = torch.randn([32, self.latent_size]).to('cuda')
            aug_x = self.decoder(z, d, yhat)
            return aug_x, torch.LongTensor(y).to('cuda')


    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, layer_chans=[1, 8, 16, 32]):

        super().__init__()

        kernel_size = 4  # (4, 4) kernel
        init_channels = 8  # initial number of filters
        image_channels = 1  # MNIST images are grayscale
        latent_dim = 16  # latent dimension for sampling
        self.num_labels = num_labels
        self.conditional = conditional
        if self.conditional:
            layer_sizes[-1] += 1
        self.enc = nn.Sequential()
        for i, (in_chan, out_chan) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.enc.add_module(
                name="L{:d}".format(i), module=nn.Conv2d(
                in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size,
                stride=2, padding=1))
            self.enc.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.fc1 = nn.Linear(288, 64)
        self.linear_means = nn.Linear(64-1-num_labels, latent_size)
        self.linear_log_var = nn.Linear(64-1-num_labels, latent_size)

    def forward(self, x, c=None):
        batch = x.shape[0]
        x = self.enc(x)
        x = x.view(batch, -1)
        hidden = self.fc1(x)
        d = hidden[:, -1]
        yhat = hidden[:, -1-self.num_labels:-1]
        means = self.linear_means(hidden[:,:-1-self.num_labels])
        log_vars = self.linear_log_var(hidden[:,:-1-self.num_labels])
        return means, log_vars, d, yhat


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, layer_chans=[32,16,8,1]):
        super().__init__()
        kernel_size = 4  # (4, 4) kernel
        self.fc1 = nn.Linear(latent_size+num_labels+1, 288)
        self.dec = nn.Sequential()
        self.conditional = conditional
        self.padding = [1, 0, 0]
        for i, (in_chan, out_chan) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.dec.add_module(
                name="L{:d}".format(i), module=nn.ConvTranspose2d(
                in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size,
                stride=2, padding=self.padding[i]))
            if i+2 < len(layer_sizes):
                self.dec.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.dec.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, d, yhat):

        z = torch.cat((z, F.softmax(yhat), d.unsqueeze(1)), dim=-1)
        hidden = self.fc1(z)
        x = self.dec(hidden.view(-1, 32, 3, 3))
        return x[:, :, 1:-1, 1:-1]
