import sys

from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu
import torch.nn.functional as F

from .latentnet import *
from .unet import *
from choices import *

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )
        
        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            # in_channels=conf.in_channels,
            in_channels=1,
            model_channels=conf.model_channels,
            out_hid_channels=256,
            out_channels=512,
            # out_hid_channels=conf.enc_out_channels,
            # out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()
        
        '''
        self.encoder2 = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            # in_channels=conf.in_channels,
            in_channels=3,
            model_channels=conf.model_channels,
            out_hid_channels=256,
            out_channels=512,
            # out_hid_channels=conf.enc_out_channels,
            # out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()
        '''
        

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()
        
        # color encoder
        self.color_encoder = torch.nn.Linear(24, 512) 
        
        '''
        # mask encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0), # for 512 size
            )
        '''
        
        # fovea encoder
        self.fovea_encoder = torch.nn.Linear(2, 512) 
        
        # od encoder
        self.od_encoder = torch.nn.Linear(2, 512)  

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond = self.encoder.forward(x)
        # return {'cond': cond}
        return cond
    
    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            
            if x is not None:
                assert len(x) == len(x_start['fovea']), f'{len(x)} != {len(x_start["fovea"])}'

            # tmp = self.encode(x_start)
            # cond = tmp['cond']
            # cond = self.encode(x_start['img'])
            
            '''
            q = self.encode1(x_start['mask'])
            k = self.encode2(x_start['img'])
            v = self.encode2(x_start['img'])
            
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
                        
            input_size = 512
            num_heads = 8
            dropout_rate = 0.1

            device = torch.device("cuda")    
            attn = torch.nn.MultiheadAttention(input_size, num_heads, dropout=dropout_rate).to(device)

            output, _ = attn(q, k, v)
            
            cond_mask = output.squeeze(1)
            '''
            
            cond_color = self.color_encoder(x_start['color'].detach())
            
            cond_mask = self.encode(x_start['mask'])
            
            cond_fovea = self.fovea_encoder(x_start['fovea'].detach())
        
            cond_od = self.od_encoder(x_start['od'].detach())

            cond_style = x_start['style'].detach()
            
            cond_disease = x_start['disease'].detach()
            
            '''
            q = cond_mask
            k = cond_od
            v = cond_od
            
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            
            input_size = 512
            num_heads = 8
            dropout_rate = 0.1
            
            device = torch.device("cuda")    
            attn = torch.nn.MultiheadAttention(input_size, num_heads, dropout=dropout_rate).to(device)            
            output, _ = attn(q, k, v)
            
            cond_mask_new = output.squeeze(1)
            '''
            
            #print(cond_mask.device.type, cond_fovea.device.type, cond_color.device.type, cond_od.device.type, cond_style.device.type, cond_disease.device.type)
            # cond = torch.cat((cond_mask, cond_color, cond_fovea), dim=1)
            # print(cond_fovea.shape, cond_style.shape, cond_disease.shape, cond_mask.shape, cond_color.shape,) 
            
            cond = cond_fovea + cond_style + cond_disease + cond_mask + cond_color + cond_od
            
            # output of cond size is [-1,512]
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
