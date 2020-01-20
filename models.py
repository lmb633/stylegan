import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Blur2d(nn.Module):
    def __init__(self, f=[1., 2., 1.], normalize=True, flip=False, stride=1):
        super().__init__()
        f = torch.tensor(f, dtype=torch.float32)
        if f.dim() == 1:
            f = f[:, None] * f[None, :]
        if normalize:
            f /= f.sum()
        if flip:
            f = torch.flip(f, [0, 1])
        self.f = f[None, None]
        self.stride = stride

    def forward(self, x):
        kernal = self.f.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernal, stride=self.stride, padding=int((self.f.size(2) - 1) / 2), groups=x.size(1))
        return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor == 1:
            return x
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1)
        x = x.expand(-1, -1, -1, self.factor, -1, self.factor)
        x = x.contiguous().view(shape[0], shape[1], shape[2] * self.factor, shape[3] * self.factor)
        return x


class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super(Downscale2d, self).__init__()
        self.factor = factor
        self.gain = gain
        self.f = [np.sqrt(gain) / factor] * factor
        self._blur2d = Blur2d(self.f, normalize=False, stride=factor)
        self.avg_pool = nn.AvgPool2d(self.factor)

    def forward(self, x):
        if self.factor == 2:
            return self._blur2d(x)
        if self.gain != 1:
            x *= self.gain
        if self.factor == 1:
            return x
        return self.avg_pool(x)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1.):
    # print(use_wscale)
    fan_in = np.prod(shape[1:])  # [fmaps_out, fmaps_in, kernel, kernel] or [out,in]
    he_std = gain / np.sqrt(fan_in)  # He init

    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul
    return torch.nn.Parameter(nn.init.normal_(torch.randn(shape), 0, init_std), requires_grad=True) * runtime_coef


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, gain=np.sqrt(2), lrmul=1., use_wscale=False):
        super(Linear, self).__init__()
        self.weight = get_weight([out_channel, in_channel], gain, lrmul=lrmul, use_wscale=use_wscale)

    def forward(self, x):
        x = F.linear(x, self.weight)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class Bias(nn.Module):
    def __init__(self, shape, lrmul=1.):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.randn(shape, requires_grad=True) * lrmul, requires_grad=True)

    def forward(self, x):
        return x + self.weight


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x2 = x ** 2
        temp = torch.mean(x2, dim=1, keepdim=True)
        return x * torch.rsqrt(temp + self.epsilon)


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x -= torch.mean(x, dim=[2, 3], keepdim=True)
        x *= torch.rsqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.epsilon)
        return x


class StyleMod(nn.Module):
    def __init__(self, dlatent_size, channels, use_wscale=False):
        super(StyleMod, self).__init__()
        self.linear = Linear(dlatent_size, channels * 2, gain=1., use_wscale=use_wscale)
        self.bias = Bias((1, channels * 2))

    def forward(self, x, dlatent):
        style = self.bias(self.linear(dlatent))
        style = style.reshape((-1, 2, x.size(1), 1, 1))
        return x * (style[:, 0] + 1.) + style[:, 1]


class ApplyNoise(nn.Module):
    def __init__(self, channel):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channel), requires_grad=True).to(device)

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.rand(x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype).to(device)
        return x + self.weight.view(1, -1, 1, 1) * noise


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, gain=np.sqrt(2), use_wscale=False):
        super(Conv2d, self).__init__()
        self.weight = get_weight([out_channel, in_channel, kernel, kernel], gain=gain, use_wscale=use_wscale)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)


class UpscaleConv2d(nn.Module):
    def __init__(self, size, in_channel, out_channel, kernel, fused_scale='auto', use_wscale=False):
        super(UpscaleConv2d, self).__init__()
        if fused_scale == 'auto':
            self.fused_scale = size >= 64
        self.weight = get_weight([out_channel, in_channel, kernel, kernel], use_wscale=use_wscale)
        self.conv = Conv2d(in_channel, out_channel, kernel, use_wscale=use_wscale)
        self.upsample = Upscale2d()

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.upsample(x))
        w = self.weight.permute(1, 0, 2, 3)
        w = F.pad(w, [1, 1, 1, 1])
        w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
        x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
        return x


class Conv2dDownscale(nn.Module):
    def __init__(self, size, in_channel, out_channel, kernel, fused_scale='auto', use_wscale=False):
        super(Conv2dDownscale, self).__init__()
        if fused_scale == 'auto':
            self.fused_scale = size >= 64
        self.weight = get_weight([out_channel, in_channel, kernel, kernel])
        self.conv = Conv2d(in_channel, out_channel, kernel, use_wscale=use_wscale)
        self.downsample = Downscale2d()

    def forward(self, x):
        if not self.fused_scale:
            return self.downsample(self.conv(x))
        w = F.pad(self.weight, [1, 1, 1, 1])
        w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
        x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
        return x


class LayerEpilogue(nn.Module):
    def __init__(self, channel, mapping_nonlinearity='lrelu', use_noise=True, use_pixel_norm=False, use_instance_norm=True, use_style=True, dlatent_size=512, use_wscale=False):
        super(LayerEpilogue, self).__init__()
        self.act = {'relu': torch.nn.ReLU(), 'lrelu': torch.nn.LeakyReLU()}[mapping_nonlinearity]
        self.use_noise = use_noise
        self.use_instance_norm = use_instance_norm
        self.use_style = use_style
        self.use_pixel_norm = use_pixel_norm

        if use_noise:
            self.noise = ApplyNoise(channel)
        self.bias = Bias((1, channel, 1, 1))
        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        if use_instance_norm:
            self.in_norm = InstanceNorm()
        if use_style:
            self.style = StyleMod(dlatent_size, channel, use_wscale=use_wscale)

    def forward(self, x, latent):
        if self.use_noise:
            x = self.noise(x)
        x = self.bias(x)
        x = self.act(x)
        if self.use_pixel_norm:
            x = self.pixel_norm(x)
        if self.use_instance_norm:
            x = self.in_norm(x)
        if self.use_style:
            x = self.style(x, latent)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channel, rgb_channel=3, kernal=1):
        super(ToRGB, self).__init__()
        self.conv = Conv2d(in_channel, out_channel=rgb_channel, kernel=kernal, use_wscale=True)
        self.bias = Bias((1, rgb_channel, 1, 1))

    def forward(self, x):
        return self.bias(self.conv(x))


class FromRGB(nn.Module):
    def __init__(self, out_channel, in_channel=3, kernal=1):
        super(FromRGB, self).__init__()
        self.conv = Conv2d(in_channel, out_channel, kernal, use_wscale=True)
        self.bias = Bias((1, out_channel, 1, 1))

    def forward(self, x):
        return self.bias(self.conv(x))


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, size, blur_filter=[1., 2., 1.], kernel=3, use_wscale=False, fused_scale='auto', **kwargs):
        super(Block, self).__init__()
        self.blur_filter = blur_filter
        self.upscale_conv = UpscaleConv2d(size, in_channel, out_channel, kernel, use_wscale=use_wscale, fused_scale=fused_scale)
        if self.blur_filter:
            self.blur = Blur2d(self.blur_filter)
        self.layer_epilogue1 = LayerEpilogue(out_channel, use_wscale=use_wscale, **kwargs)
        self.conv = Conv2d(out_channel, out_channel, kernel, use_wscale=use_wscale)
        self.layer_epilogue2 = LayerEpilogue(out_channel, use_wscale=use_wscale, **kwargs)

    def forward(self, x, latent):
        x = self.upscale_conv(x)
        if self.blur_filter:
            x = self.blur(x)
        x = self.layer_epilogue1(x, latent[:, 0].squeeze(1))
        x = self.conv(x)
        x = self.layer_epilogue2(x, latent[:, 1].squeeze(1))
        return x


class InitBlock(nn.Module):
    def __init__(self, const_input_layer, channel, gain, use_wscale=False, dlatent_size=512, **kwargs):
        super(InitBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.channel = channel
        self.init_const = torch.nn.Parameter(torch.ones([1, channel, 4, 4]), requires_grad=True)
        self.lin = Linear(dlatent_size, channel * 16, gain=gain / 4, use_wscale=use_wscale)
        self.layer_epilogue1 = LayerEpilogue(channel, use_wscale=use_wscale, **kwargs)
        self.conv = Conv2d(channel, channel, 3, use_wscale=use_wscale)
        self.layer_epilogue2 = LayerEpilogue(channel, use_wscale=use_wscale, **kwargs)

    def forward(self, latent):
        batchsize = latent.size(0)
        if self.const_input_layer:
            x = self.init_const.expand(batchsize, -1, -1, -1)
        else:
            x = self.lin(latent[:, 0].squeeze(1))
            x = x.view(batchsize, self.channel, 4, 4)
        x = self.layer_epilogue1(x, latent[:, 0].squeeze(1))
        x = self.conv(x)
        x = self.layer_epilogue2(x, latent[:, 1].squeeze(1))
        return x


class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features,
                       c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


class LastLayer(nn.Module):
    def __init__(self, mbstd_group_size, mbstd_num_features, in_channel, out_channel, activation_layer,
                 kernel=3, gain=np.sqrt(2), use_wscale=True, label_size=0, resolution=4, last_gain=1):
        super(LastLayer, self).__init__()
        self.mbstd_group_size = mbstd_group_size
        if mbstd_group_size > 1:
            self.mb_std_layer = StddevLayer(mbstd_group_size, mbstd_num_features)
        self.conv2d = Conv2d(in_channel + 1, out_channel, kernel, gain=gain, use_wscale=use_wscale)
        self.bias1 = Bias((1, out_channel, 1, 1))
        self.act1 = activation_layer
        self.lin1 = Linear(out_channel * resolution * resolution, in_channel, gain=gain)
        self.bias2 = Bias((1, in_channel))
        self.act2 = activation_layer
        self.lin2 = Linear(in_channel, max(label_size, 1), gain=last_gain)
        self.bias3 = Bias((1, max(label_size, 1)))

    def forward(self, x):
        if self.mbstd_group_size > 1:
            x = self.mb_std_layer(x)
        print(x.shape)
        x = self.act1(self.bias1(self.conv2d(x)))
        x = self.lin1(x.view(x.size(0), -1))
        x = self.act2(self.bias2(x))
        x = self.bias3(self.lin2(x))
        return x


class GMapping(nn.Module):
    def __init__(self,
                 latent_size=512,  # Latent vector (Z) dimensionality.
                 label_size=0,  # Label dimensionality, 0 if no labels.
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 dlatent_broadcast=0,  # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
                 mapping_layers=8,  # Number of mapping layers.
                 mapping_lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'.
                 use_wscale=True,  # Enable equalized learning rate?
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 **_kwargs):  # Ignore unrecognized keyword args.
        super(GMapping, self).__init__()
        self.act, self.gain = {'relu': (torch.nn.ReLU(), np.sqrt(2)), 'lrelu': (torch.nn.LeakyReLU(), np.sqrt(2))}[mapping_nonlinearity]
        self.label_size = label_size
        self.broadcast = dlatent_broadcast

        if label_size:
            self.label_emb = nn.Embedding(label_size, latent_size)
            latent_size *= 2
        self.lin = nn.Sequential()

        for i in range(mapping_layers):
            if i == 0:
                self.lin.add_module('lin{0}'.format(i + 1), Linear(latent_size, dlatent_size, self.gain, mapping_lrmul, use_wscale=use_wscale))
            else:
                self.lin.add_module('lin{0}'.format(i + 1), Linear(dlatent_size, dlatent_size, self.gain, mapping_lrmul, use_wscale=use_wscale))
            self.lin.add_module('bias{0}'.format(i + 1), Bias((1, dlatent_size)))
            self.lin.add_module('act{0}'.format(i + 1), self.act)

        self.norm = normalize_latents
        self.normlization = PixelNorm()

    def forward(self, x, label_in=None):
        if self.label_size:
            label = self.label_emb(label_in)
            x = torch.cat([x, label], dim=1)
        if self.norm:
            x = self.normlization(x)
        x = self.lin(x)
        print(x.shape)
        if self.broadcast:
            x = x.unsqueeze(1).expand(-1, self.broadcast, -1)
        return x


class GSynthesis(nn.Module):
    def __init__(self,
                 kernel=3,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 num_channels=3,  # Number of output color channels.
                 resolution=1024,  # Output resolution.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 use_styles=True,  # Enable style inputs?
                 const_input_layer=True,  # First layer is a learned constant?
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'
                 use_wscale=True,  # Enable equalized learning rate?
                 fused_scale='auto',  # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
                 blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 **_kwargs):  # Ignore unrecognized keyword args.):
        super(GSynthesis, self).__init__()
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        # num_layers = resolution_log2 * 2 - 2
        # num_style = num_layers if use_styles else 1
        self.structure = structure
        act, gain = {'relu': (torch.nn.ReLU(), np.sqrt(2)), 'lrelu': (torch.nn.LeakyReLU(), np.sqrt(2))}[nonlinearity]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.first_block = InitBlock(const_input_layer, nf(1), gain / 4, use_wscale, dlatent_size, **_kwargs)
        self.block = []
        self.rgb_block = []
        self.rgb_block.append(ToRGB(nf(1), num_channels))
        for layer in range(3, resolution_log2 + 1):
            in_channel = nf(layer - 2)
            out_channel = nf(layer - 1)
            size = 4 * 2 ** (layer - 2)
            self.block.append(Block(in_channel, out_channel, size, blur_filter, kernel, use_wscale, fused_scale=fused_scale, **_kwargs))
            self.rgb_block.append(ToRGB(out_channel, num_channels, kernel))

    def forward(self, dlatents, depth=0, alpha=1., label_in=None):
        x = self.first_block(dlatents[:, 0:2])
        if self.structure == 'fixed':
            for i, block in enumerate(self.block):
                x = block(x, dlatents[:, 2 * (i + 1):2 * (i + 2)])
            image_out = self.rgb_block[-1](x)
        if self.structure == 'linear':
            for i in range(depth - 1):
                x = self.block[i](x, dlatents[:, 2 * (i + 1):2 * (i + 2)])
            residual = self.rgb_block[depth - 1](Upscale2d()(x))
            straight = self.rgb_block[depth](self.block[depth - 1](x, dlatents[:, 2 * depth:2 * (depth + 1)]))
            image_out = straight * alpha + (1 - alpha) * residual
        return image_out


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


class Gererator(nn.Module):
    def __init__(self, resolution, training=True, latent_size=512, dlatent_size=512, label_size=0,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995, style_mixing_prob=0.9, **kwargs):
        super(Gererator, self).__init__()
        self.training = training
        self.style_mixing_prob = style_mixing_prob
        self.lin_layer = (int(np.log2(resolution)) - 1) * 2

        self.g_mapping = GMapping(latent_size, label_size, dlatent_size, self.lin_layer, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)
        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size), max_layer=truncation_cutoff, threshold=truncation_psi, beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, x, latent_in, depth, alpha, label_in=None):
        dlatent_in = self.g_mapping(latent_in)
        if self.training:
            if self.truncation is not None:
                self.truncation.update(dlatent_in[0, 0].detach())
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latent_in2 = torch.randn(latent_in.shape).to(device)
                dlatent_in2 = self.g_mapping(latent_in2)
                layer_idx = torch.from_numpy(np.arange(self.lin_layer)[np.newaxis, :, np.newaxis]).to(device)
                cur_layer = (depth + 1) * 2
                mixing_cutoff = random.randint(1, cur_layer) if random.random() < self.style_mixing_prob else cur_layer
                dlatent_in = torch.where(layer_idx < mixing_cutoff, dlatent_in, dlatent_in2)
            if self.truncation is not None:
                dlatent_in = self.truncation(dlatent_in)
        image = self.g_synthesis(dlatent_in, depth, alpha)
        return image


class Discriminiator(nn.Module):
    def __init__(self, resolution, kernel=3, label_size=0, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, fused_scale='suto', mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='linear', **kwargs):
        super(Discriminiator, self).__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_feature = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1
        act, gain = {'relu': (torch.nn.ReLU(), np.sqrt(2)), 'lrelu': (torch.nn.LeakyReLU(), np.sqrt(2))}[nonlinearity]

        self.block = []
        self.from_rgb_block = []
        for layer in range(resolution_log2, 2, -1):
            in_channel = nf(layer - 1)
            out_channel = nf(layer - 2)
            size = 2 ** layer

            self.block.append(Block(in_channel, out_channel, size, blur_filter, kernel, use_wscale, fused_scale=fused_scale, **kwargs))
            self.from_rgb_block.append(FromRGB(out_channel=in_channel))
        self.from_rgb_block.append(FromRGB(nf(2)))
        self.last_layer = LastLayer(mbstd_group_size, mbstd_num_features, out_channel, out_channel, act, kernel, gain, use_wscale, label_size=label_size)

    def forward(self, x, depth, alpha=1., labels_in=None):
        if self.structure == 'fixed':
            x = self.from_rgb_block[0](x)
            for i, block in enumerate(self.block):
                x = block(x)
            x = self.last_layer(x)
        if self.structure == 'linear':
            if depth > 0:
                residual = self.from_rgb_block[self.depth - depth](Downscale2d()(x))
                straight = self.block[self.depth - depth - 1](self.from_rgb_block[self.depth - depth - 1](x))
                x = alpha * straight + (1 - alpha) * residual
                for block in self.block[self.depth - depth:]:
                    x = block(x)
            else:
                x = self.block[self.from_rgb_block[-1](x)]
            x = self.last_layer(x)
        return x


if __name__ == '__main__':
    import tensorflow as tf

    indata = torch.randn((2, 512, 4, 4))
    last = LastLayer(4, 1, 512, 512, nn.ReLU())
    out = last(indata)
    print(out.shape)


    # indata = torch.randn((2, 64, 112, 112))
    # latent = torch.randn((2, 512))
    # latent_in = torch.randn((1, 512))
    # g_mapping = GMapping(512, dlatent_broadcast=18)
    # dlatent = g_mapping(latent_in)
    # print(dlatent.shape)
    # g_synthesis = GSynthesis(dlatent_size=512, resolution=1024, structure='linear')
    # out = g_synthesis(dlatent, 0)
    # print(out.shape)


    # style = StyleMod(512, 64)
    # out = style(indata, latent)
    # print(out.shape)
    # layer = LayerEpilogue(64)
    # out = layer(indata, latent)
    # print(out.shape)
    # conv = Conv2d(64, 128, 3)
    # out = conv(indata)
    # print(out.shape)

    # indata = torch.randn((2, 32, 112, 112))
    # upconv = UpscaleConv2d(112, 32, 64, 3)
    # out = upconv(indata)
    # print(out.shape)

    # indata = torch.randn((2, 128, 112, 112))
    # downconv = Conv2dDownscale(112, 128, 16, 3)
    # out = downconv(indata)
    # print(out.shape)
    #
    # mapping_nonlinearity='lrelu', use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm, use_style=use_style, dlatent_size=dlatent_size, **kwargs

    # indata = torch.randn((2, 128, 112, 112))
    # latent = torch.randn((2, 2, 512))
    # block = Block(128, 32, 112, use_wscale=True)
    # out = block(indata, latent)
    # print(out.shape)
    #
    # init = InitBlock(True, 512, 1, True)
    # out = init(latent)
    # print(out.shape)




# indata = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
# print(indata.shape)
# model = Upscale2d(factor=2, gain=1)
# out = model(indata)
# print(out)
# avg = nn.AvgPool2d(2, padding=1)
# out = avg(indata)
# print(out)
# x = torch.randn((12, 512))
# indata = torch.randn((64, 512))
# label = torch.LongTensor(np.zeros((64)))
# fc = GMapping(latent_size=512, label_size=4, dlatent_size=512, dlatent_broadcast=18, mapping_layers=8)
# out = fc(indata, label)
# print(out.shape)
