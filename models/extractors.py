import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
import numpy as np


class TDNN(nn.Module):
    def __init__(
        self,
        input_dim=23,
        output_dim=512,
        context_size=5,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout_p=0.0,
        padding=0,
    ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding

        self.kernel = nn.Conv1d(
            self.input_dim,
            self.output_dim,
            self.context_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.nonlinearity = nn.LeakyReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        """
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        """

        _, _, d = x.shape
        assert (
            d == self.input_dim
        ), "Input dimension was wrong. Expected ({}), got ({})".format(
            self.input_dim, d
        )

        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:
            x = self.bn(x)
        return x.transpose(1, 2)


class StatsPool(nn.Module):
    def __init__(self, floor=1e-10, bessel=False, T_dim=1):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel
        self.T_dim = T_dim

    def forward(self, x):
        means = torch.mean(x, dim=self.T_dim)
        t = x.shape[self.T_dim]
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(self.T_dim)
        numerator = torch.sum(residuals**2, dim=self.T_dim)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor) / t)
        x = torch.cat([means, stds], dim=-1)
        return x


class ETDNN(nn.Module):
    def __init__(
        self,
        features_per_frame=80,
        hidden_features=512,
        embed_features=256,
        dropout_p=0.0,
        batch_norm=True,
    ):
        super(ETDNN, self).__init__()
        self.features_per_frame = features_per_frame
        self.hidden_features = hidden_features
        self.embed_features = embed_features

        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {"dropout_p": dropout_p, "batch_norm": self.batch_norm}
        self.nl = nn.LeakyReLU()

        self.frame1 = TDNN(
            input_dim=self.features_per_frame,
            output_dim=self.hidden_features,
            context_size=5,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame2 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame3 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=3,
            dilation=2,
            **tdnn_kwargs
        )
        self.frame4 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame5 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=3,
            dilation=3,
            **tdnn_kwargs
        )
        self.frame6 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame7 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=3,
            dilation=4,
            **tdnn_kwargs
        )
        self.frame8 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame9 = TDNN(
            input_dim=self.hidden_features,
            output_dim=self.hidden_features * 3,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )

        self.tdnn_list = nn.Sequential(
            self.frame1,
            self.frame2,
            self.frame3,
            self.frame4,
            self.frame5,
            self.frame6,
            self.frame7,
            self.frame8,
            self.frame9,
        )
        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.hidden_features * 6, self.embed_features)

    def forward(self, x):
        x = self.tdnn_list(x)
        x = self.statspool(x)
        x = self.fc_embed(x)
        return x


class XTDNN(nn.Module):
    def __init__(
        self,
        features_per_frame=30,
        final_features=1500,
        embed_features=512,
        dropout_p=0.0,
        batch_norm=True,
    ):
        super(XTDNN, self).__init__()
        self.features_per_frame = features_per_frame
        self.final_features = final_features
        self.embed_features = embed_features
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {"dropout_p": dropout_p, "batch_norm": self.batch_norm}

        self.frame1 = TDNN(
            input_dim=self.features_per_frame,
            output_dim=512,
            context_size=5,
            dilation=1,
            **tdnn_kwargs
        )
        self.frame2 = TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=2, **tdnn_kwargs
        )
        self.frame3 = TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=3, **tdnn_kwargs
        )
        self.frame4 = TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs
        )
        self.frame5 = TDNN(
            input_dim=512,
            output_dim=self.final_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )

        self.tdnn_list = nn.Sequential(
            self.frame1, self.frame2, self.frame3, self.frame4, self.frame5
        )
        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.final_features * 2, self.embed_features)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embed_features)
        self.drop_embed = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.tdnn_list(x)
        x = self.statspool(x)
        x = self.fc_embed(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        return x


class XTDNN_ILayer(nn.Module):
    def __init__(self, features_per_frame=30, dropout_p=0.0, batch_norm=True):
        super().__init__()
        self.features_per_frame = features_per_frame
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {"dropout_p": dropout_p, "batch_norm": self.batch_norm}

        self.frame1 = TDNN(
            input_dim=self.features_per_frame,
            output_dim=512,
            context_size=5,
            dilation=1,
            **tdnn_kwargs
        )

    def forward(self, x):
        x = self.frame1(x)
        return x


class XTDNN_OLayer(nn.Module):
    def __init__(
        self, final_features=1500, embed_features=512, dropout_p=0.0, batch_norm=True
    ):
        super().__init__()
        self.final_features = final_features
        self.embed_features = embed_features
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {"dropout_p": dropout_p, "batch_norm": self.batch_norm}

        self.frame2 = TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=2, **tdnn_kwargs
        )
        self.frame3 = TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=3, **tdnn_kwargs
        )
        self.frame4 = TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, **tdnn_kwargs
        )
        self.frame5 = TDNN(
            input_dim=512,
            output_dim=self.final_features,
            context_size=1,
            dilation=1,
            **tdnn_kwargs
        )

        self.statspool = StatsPool()

        self.fc_embed = nn.Linear(self.final_features * 2, self.embed_features)
        self.nl_embed = nn.LeakyReLU()
        self.bn_embed = nn.BatchNorm1d(self.embed_features)
        self.drop_embed = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)
        x = self.statspool(x)
        x = self.fc_embed(x)
        x = self.nl_embed(x)
        x = self.bn_embed(x)
        x = self.drop_embed(x)
        return x


class DenseReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseReLU, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nl = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.nl(x)
        if len(x.shape) > 2:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.bn(x)
        return x


class FTDNNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bottleneck_dim,
        context_size=2,
        dilations=None,
        paddings=None,
        alpha=0.0,
    ):
        """
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        """
        super(FTDNNLayer, self).__init__()
        paddings = [1, 1, 1] if not paddings else paddings
        dilations = [2, 2, 2] if not dilations else dilations
        kwargs = {"bias": False}
        self.factor1 = nn.Conv1d(
            in_dim,
            bottleneck_dim,
            context_size,
            padding=paddings[0],
            dilation=dilations[0],
            **kwargs
        )
        self.factor2 = nn.Conv1d(
            bottleneck_dim,
            bottleneck_dim,
            context_size,
            padding=paddings[1],
            dilation=dilations[1],
            **kwargs
        )
        self.factor3 = nn.Conv1d(
            bottleneck_dim,
            out_dim,
            context_size,
            padding=paddings[2],
            dilation=dilations[2],
            **kwargs
        )
        self.reset_parameters()
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = SharedDimScaleDropout(alpha=alpha, dim=1)

    def forward(self, x):
        """input (batch_size, seq_len, in_dim)"""
        assert x.shape[-1] == self.factor1.weight.shape[1]
        x = self.factor1(x.transpose(1, 2))
        x = self.factor2(x)
        x = self.factor3(x)
        x = self.nl(x)
        x = self.bn(x).transpose(1, 2)
        x = self.dropout(x)
        return x

    def step_semi_orth(self):
        with torch.no_grad():
            factor1_M = self.get_semi_orth_weight(self.factor1)
            factor2_M = self.get_semi_orth_weight(self.factor2)
            self.factor1.weight.copy_(factor1_M)
            self.factor2.weight.copy_(factor2_M)

    def reset_parameters(self):
        # Standard dev of M init values is inverse of sqrt of num cols
        nn.init._no_grad_normal_(
            self.factor1.weight, 0.0, self.get_M_shape(self.factor1.weight)[1] ** -0.5
        )
        nn.init._no_grad_normal_(
            self.factor2.weight, 0.0, self.get_M_shape(self.factor2.weight)[1] ** -0.5
        )

    def orth_error(self):
        factor1_err = self.get_semi_orth_error(self.factor1).item()
        factor2_err = self.get_semi_orth_error(self.factor2).item()
        return factor1_err + factor2_err

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1] * orig_shape[2]
            ).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.999
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP / trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return (
                updated.reshape(*orig_shape)
                if mshape[0] > mshape[1]
                else updated.T.reshape(*orig_shape)
            )

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1] * orig_shape[2], orig_shape[0])

    @staticmethod
    def get_semi_orth_error(conv1dlayer):
        with torch.no_grad():
            orig_shape = conv1dlayer.weight.shape
            M = conv1dlayer.weight.reshape(orig_shape[0], orig_shape[1] * orig_shape[2])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)
            scale2 = torch.sqrt(trace_PP / trace_P) ** 2
            update = P - (torch.matrix_power(P, 0) * scale2)
            return torch.norm(update, p="fro")


class FTDNN(nn.Module):
    def __init__(self, in_dim=30, embedding_dim=512):
        """
        The FTDNN architecture from
        "State-of-the-art speaker recognition with neural network embeddings in
        NIST SRE18 and Speakers in the Wild evaluations"
        https://www.sciencedirect.com/science/article/pii/S0885230819302700
        """
        super(FTDNN, self).__init__()

        self.layer01 = TDNN(input_dim=in_dim, output_dim=512, context_size=5, padding=2)
        self.layer02 = FTDNNLayer(
            512, 1024, 256, context_size=2, dilations=[2, 2, 2], paddings=[1, 1, 1]
        )
        self.layer03 = FTDNNLayer(
            1024, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0]
        )
        self.layer04 = FTDNNLayer(
            1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1]
        )
        self.layer05 = FTDNNLayer(
            2048, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0]
        )
        self.layer06 = FTDNNLayer(
            1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1]
        )
        self.layer07 = FTDNNLayer(
            3072, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1]
        )
        self.layer08 = FTDNNLayer(
            1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1]
        )
        self.layer09 = FTDNNLayer(
            3072, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0]
        )
        self.layer10 = DenseReLU(1024, 2048)

        self.layer11 = StatsPool()

        self.layer12 = DenseReLU(4096, embedding_dim)

    def forward(self, x):
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        skip_5 = torch.cat([x_4, x_3], dim=-1)
        x = self.layer05(skip_5)
        x_6 = self.layer06(x)
        skip_7 = torch.cat([x_6, x_4, x_2], dim=-1)
        x = self.layer07(skip_7)
        x_8 = self.layer08(x)
        skip_9 = torch.cat([x_8, x_6, x_4], dim=-1)
        x = self.layer09(skip_9)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.0
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors


class SharedDimScaleDropout(nn.Module):
    def __init__(self, alpha: float = 0.5, dim=1):
        """
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        """
        super(SharedDimScaleDropout, self).__init__()
        if alpha > 0.5 or alpha < 0:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha
        self.dim = dim
        self.register_buffer("mask", torch.tensor(0.0))

    def forward(self, X):
        if self.training:
            if self.alpha != 0.0:
                # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
                tied_mask_shape = list(X.shape)
                tied_mask_shape[self.dim] = 1
                repeats = [
                    1 if i != self.dim else X.shape[self.dim]
                    for i in range(len(X.shape))
                ]
                return X * self.mask.repeat(tied_mask_shape).uniform_(
                    1 - 2 * self.alpha, 1 + 2 * self.alpha
                ).repeat(repeats)
                # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return X


class AttStatsPooling(nn.Module):
    def __init__(self, in_dim, att_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, att_dim, kernel_size=1)
        self.nl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(att_dim)
        self.linear2 = nn.Conv1d(att_dim, 1, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(in_dim * 2)

    def forward(self, h):
        """
        input: (batch_size, in_dim, seq_len)
        output: (batch_size, in_dim*2)
        """
        e = self.linear1(h)
        e = self.nl(e)
        e = self.bn1(e)
        e = self.linear2(e).squeeze(1)

        alpha = F.softmax(e, dim=-1).unsqueeze(1)
        mu = (alpha * h).sum(dim=-1).unsqueeze(-1)

        sigma = (alpha * h**2) - mu**2
        sigma = torch.sqrt(torch.clamp(sigma.sum(dim=-1), min=1e-4))
        x = torch.cat([mu.squeeze(-1), sigma], dim=1)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.se = nn.Sequential(
            self.avg, self.linear1, self.relu, self.bn, self.linear2, self.sigmoid
        )

    def forward(self, input):
        """
        input: (batch_size, in_dim, seq_len)
        output: same as input
        """
        x = self.se(input)
        x = input * x

        return x


class ConvReluBN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, padding=0, dilation=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Res2NetBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size=3, dilation=1, se_bottleneck=128, scale=4
    ):
        super().__init__()

        self.width = int(np.floor(out_dim / scale))
        self.scale = scale

        self.convrelubn1 = ConvReluBN(
            in_dim, self.width * scale, kernel_size=1, padding=0, dilation=1
        )
        self.relu = nn.ReLU()

        convs = []
        bns = []

        padding = int(np.floor(kernel_size / 2) * dilation)

        for i in range(scale - 1):
            convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            bns.append(nn.BatchNorm1d(self.width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.convrelubn2 = ConvReluBN(
            self.width * scale, out_dim, kernel_size=1, padding=0, dilation=1
        )
        self.se = SEBlock(out_dim, se_bottleneck)

    def forward(self, x):
        """
        input: (batch_size, in_dim, seq_len)
        """
        residual = x

        out = self.convrelubn1(x)

        split = torch.split(out, self.width, 1)
        for i in range(self.scale - 1):
            if i == 0:
                mini = split[0]
            else:
                mini = mini + split[i]

            mini = self.convs[i](mini)
            mini = self.relu(mini)
            mini = self.bns[i](mini)

            if i == 0:
                out = mini
            else:
                out = torch.cat([out, mini], 1)

        out = torch.cat([out, split[-1]], 1)

        out = self.convrelubn1(out)

        out = self.se(out)
        out += residual

        return out


class ECAPATDNN(nn.Module):
    def __init__(
        self,
        in_dim=30,
        channels=512,
        att_dim=128,
        embed_dim=192,
        T_dim=1,
        globalpool=False,
    ):
        super().__init__()
        self.T_dim = T_dim
        self.globalpool = globalpool

        self.crb1 = ConvReluBN(in_dim, channels, kernel_size=5, dilation=1)

        self.r2block1 = Res2NetBlock(
            channels, channels, kernel_size=3, dilation=2, scale=8
        )
        self.r2block2 = Res2NetBlock(
            channels, channels, kernel_size=3, dilation=3, scale=8
        )
        self.r2block3 = Res2NetBlock(
            channels, channels, kernel_size=3, dilation=4, scale=8
        )

        self.crb2 = ConvReluBN(channels, channels, kernel_size=1, dilation=1)

        self.pooling = AttStatsPooling(channels, att_dim=att_dim)
        if self.globalpool:
            self.gpooling = StatsPool(T_dim=self.T_dim)
            self.final_linear = nn.Linear(channels * 4, embed_dim)
        else:
            self.final_linear = nn.Linear(channels * 2, embed_dim)

        self.final_bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        if self.T_dim != -1:
            x = x.transpose(-1, self.T_dim)
        x = self.crb1(x)

        out_block1 = self.r2block1(x)
        out_block2 = self.r2block2(out_block1)
        out_block3 = self.r2block3(out_block2)

        # cat_out = torch.cat([out_block1, out_block2, out_block3], dim=1)
        out = out_block1 + out_block2 + out_block3
        out = self.crb2(out)
        pooled = self.pooling(out)
        if self.globalpool:
            gpooled = self.gpooling(out)
            out = torch.cat([pooled, gpooled], 1)
        else:
            out = pooled
        out = self.final_linear(out)
        out = self.final_bn(out)
        return out
