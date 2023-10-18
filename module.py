import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Tuple
from submodule import StackConvNorm, StackSubPixelNorm, \
    StackMLP, ConvLSTMCell
from torch.distributions import RelaxedBernoulli, Normal
from utils import Tensor
from utils import assert_shape
from utils import build_grid
from utils import conv_transpose_out_shape

class ImgEncoder(nn.Module):

    def __init__(self, args: Any):
        super(ImgEncoder, self).__init__()

        self.args = args

        modules = []
        channels = self.args.arch.in_channels
        # Build Encoder
        for h_dim in self.args.arch.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.args.arch.kernel_size,
                        stride=1,
                        padding=self.args.arch.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.cnn_layers = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.args.arch.in_channels, self.args.arch.hidden_dims[-1], self.args.arch.resolution)
        self.cnn_out_layer = nn.Sequential(
            nn.Linear(self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        cnn_out = self.cnn_layers(x)
        cnn_out = self.encoder_pos_embedding(cnn_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        cnn_out = torch.flatten(cnn_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.cnn_out_layer(cnn_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        return cnn_out

class GlobalEncoder(nn.Module):
    def __init__(self, args: Any):
        super(GlobalEncoder, self).__init__()
        self.args = args
        self.fc_mu = nn.Sequential(
            nn.LayerNorm(self.args.arch.hidden_dims[-1]*self.args.arch.resolution[0]*self.args.arch.resolution[1]),
            nn.Linear(self.args.arch.hidden_dims[-1]*self.args.arch.resolution[0]*self.args.arch.resolution[1], \
                      self.args.arch.global_latent_dim),
            nn.ReLU(True),
            nn.Linear(self.args.arch.global_latent_dim, self.args.arch.global_latent_dim)
        )
        self.fc_std = nn.Sequential(
            nn.LayerNorm(self.args.arch.hidden_dims[-1] * self.args.arch.resolution[0] * self.args.arch.resolution[1]),
            nn.Linear(self.args.arch.hidden_dims[-1] * self.args.arch.resolution[0] * self.args.arch.resolution[1], \
                      self.args.arch.global_latent_dim),
            nn.ReLU(True),
            nn.Linear(self.args.arch.global_latent_dim, self.args.arch.global_latent_dim)
        )

    def forward(self, img_enc):
        z_g_mu = self.fc_mu(img_enc)
        z_g_std = self.fc_std(img_enc)
        z_g_std = F.softplus(z_g_std) + 1e-8

        return z_g_mu, z_g_std

class LocalLatentEncoder(nn.Module):

    def __init__(self, args: Any):
        super(LocalLatentEncoder, self).__init__()
        self.args = args

        self.slot_attention = SlotAttention(
            in_features=self.args.arch.hidden_dims[-1],
            num_iterations=self.args.arch.num_iterations,
            num_slots=self.args.arch.num_slots,
            slot_size=self.args.arch.slot_size,
            mlp_hidden_size=128,
        )
        self.fc_mu = nn.Linear(self.args.arch.slot_size, self.args.arch.latent_dim)
        self.fc_std = nn.Linear(self.args.arch.slot_size, self.args.arch.latent_dim)


    def forward(self, img_enc: torch.Tensor, slot_init) -> List:
        """
        :param img_enc: (bs, height*width, filter_size)
        :return:
        """
        batch_size = img_enc.shape[0]
        slots, slots_init = self.slot_attention(img_enc, slot_init)
        assert_shape(slots.size(), (batch_size, self.args.arch.num_slots, self.args.arch.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].

        q_zs_given_x_mean = torch.empty_like(slots)
        q_zs_given_x_std = torch.empty_like(slots)
        # batch_size, num_slots, slot_size = slots.shape

        for i in range(self.args.arch.num_slots):
            q_zs_given_x_mean[:, i, :] = self.fc_mu(slots[:, i, :])
            q_zs_given_x_std[:, i, :] = self.fc_std(slots[:, i, :])

        q_zs_given_x_std = F.softplus(q_zs_given_x_std) + 1e-8


        return q_zs_given_x_mean, q_zs_given_x_std, slots_init

class GlobalToFeature(nn.Module):
    def __init__(self, args: Any):
        super(GlobalToFeature, self).__init__()
        self.args = args
        self.fc = nn.Linear(self.args.arch.global_latent_dim, \
                            self.args.arch.decoder_resolution[0]*self.args.arch.decoder_resolution[1]*self.args.arch.hidden_dims[-1])

        modules = []
        for i in range(len(self.args.arch.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.args.arch.hidden_dims[i],
                        self.args.arch.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1], kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1], kernel_size=3, stride=1, padding=1, output_padding=0, ),
            )
        )
        self.dcnn_layers = nn.Sequential(*modules)

        self.cnn_out_layer = nn.Sequential(
            nn.Linear(self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(self.args.arch.hidden_dims[-1], self.args.arch.hidden_dims[-1]),
        )

    def forward(self, z_g):
        # zg has shape: [batch_size, global_latent_dim]
        feature = self.fc(z_g)
        feature = feature.view(z_g.shape[0], self.args.arch.hidden_dims[-1], self.args.arch.decoder_resolution[0], self.args.arch.decoder_resolution[1])
        feature = self.dcnn_layers(feature)
        feature = torch.flatten(feature, start_dim=2, end_dim=3)
        feature = feature.permute(0, 2, 1)
        feature = self.cnn_out_layer(feature)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]
        return feature


class GlobalToSlots(nn.Module):
    def __init__(self, args: Any):
        super(GlobalToSlots, self).__init__()
        self.args = args
        self.zg_to_zs_mu = nn.Sequential(nn.LayerNorm(self.args.arch.global_latent_dim),
                                      nn.Linear(self.args.arch.global_latent_dim, self.args.arch.slot_size*self.args.arch.num_slots),
                                      nn.ReLU(True),
                                      nn.Linear(self.args.arch.slot_size*self.args.arch.num_slots, self.args.arch.slot_size*self.args.arch.num_slots))
        self.zg_to_zs_std = nn.Sequential(nn.LayerNorm(self.args.arch.global_latent_dim),
                                         nn.Linear(self.args.arch.global_latent_dim,
                                                   self.args.arch.slot_size * self.args.arch.num_slots),
                                         nn.ReLU(True),
                                         nn.Linear(self.args.arch.slot_size * self.args.arch.num_slots,
                                                   self.args.arch.slot_size * self.args.arch.num_slots))

    def forward(self, z_g):
        # zg has shape: [batch_size, global_latent_dim]
        p_zs_given_zg_mean = self.zg_to_zs_mu(z_g)
        p_zs_given_zg_mean = p_zs_given_zg_mean.reshape(z_g.shape[0], self.args.arch.num_slots, self.args.arch.slot_size)
        # p_zs_given_zg_mean has shape: [batch_size, num_slots, slot_size]

        p_zs_given_zg_std = self.zg_to_zs_std(z_g)
        p_zs_given_zg_std = p_zs_given_zg_std.reshape(z_g.shape[0], self.args.arch.num_slots, self.args.arch.slot_size)
        # p_zs_given_zg_std has shape: [batch_size, num_slots, slot_size]
        p_zs_given_zg_std = F.softplus(p_zs_given_zg_std) + 1e-8

        return p_zs_given_zg_mean, p_zs_given_zg_std

class LocalSampler(nn.Module):

    def __init__(self, args: Any):
        super(LocalSampler, self).__init__()
        self.args = args

    def forward(self, ss: List, phase_use_mode: bool = False) -> Tuple:

        p_z_mean, p_z_std = ss

        z = Normal(p_z_mean, p_z_std).rsample()

        return z

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.resolution = args.arch.resolution
        self.in_channels = args.arch.in_channels
        self.hidden_dims = args.arch.hidden_dims
        self.out_features = args.arch.hidden_dims[-1]
        self.decoder_resolution = args.arch.decoder_resolution
        self.decoder_out_dim = args.arch.in_channels+1

        modules = []

        in_size = self.decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, self.decoder_out_dim, kernel_size=3, stride=1, padding=1, output_padding=0, ),
            )
        )

        assert_shape(self.resolution, (out_size, out_size), message="")

        self.dcnn_layers = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

    def forward(self, x, batch_size, num_slots, num_channels, height, width):
        # shape of x: batch_size * num_slots, slot_size, decoder_resolution[0], decoder_resolution[1]

        out = self.decoder_pos_embedding(x)
        out = self.dcnn_layers(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        #assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks

class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj

class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor, slots_init):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_init == None:
            slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots, slots_init

class StructDRAW(nn.Module):

    def __init__(self, args):
        super(StructDRAW, self).__init__()
        self.args = args

        self.p_global_decoder_net = StackMLP(
            self.args.z.z_global_dim,
            self.args.arch.mlp.p_global_decoder_filters,
            norm_act_final=True
        )

        rnn_enc_inp_dim = self.args.arch.img_enc_dim * 2 + \
                          self.args.arch.structdraw.rnn_decoder_hid_dim

        rnn_dec_inp_dim = self.args.arch.mlp.p_global_decoder_filters[-1] // \
                          (self.args.arch.num_cell ** 2)

        rnn_dec_inp_dim += self.args.arch.structdraw.hid_to_dec_filters[-1]

        self.rnn_enc = ConvLSTMCell(
            input_dim=rnn_enc_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_encoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.rnn_dec = ConvLSTMCell(
            input_dim=rnn_dec_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_decoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.p_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.mlp.p_global_encoder_filters,
            norm_act_final=False
        )

        self.q_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_encoder_hid_dim,
            self.args.arch.mlp.q_global_encoder_filters,
            norm_act_final=False
        )

        self.hid_to_dec_net = StackConvNorm(
            self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.structdraw.hid_to_dec_filters,
            self.args.arch.structdraw.hid_to_dec_kernel_sizes,
            self.args.arch.structdraw.hid_to_dec_strides,
            self.args.arch.structdraw.hid_to_dec_groups,
            norm_act_final=False
        )

        self.register_buffer('dec_step_0', torch.zeros(1, self.args.arch.structdraw.hid_to_dec_filters[-1],
                                                       self.args.arch.num_cell, self.args.arch.num_cell))

        self.enc = StackConvNorm(
            self.args.arch.latent_dim,
            self.args.arch.conv.img_encoder_filters,
            self.args.arch.conv.img_encoder_kernel_sizes,
            self.args.arch.conv.img_encoder_strides,
            self.args.arch.conv.img_encoder_groups,
            norm_act_final=True
        )

        self.dec = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
            )

    def forward(self, x: torch.Tensor, phase_generation: bool = False,
                generation_from_step: Any = None, z_global_predefine: Any = None) -> Tuple:
        """
        :param x: [batch_size, height*width, filter_size]
        :return:
        """
        bs = x.size(0)
        if phase_generation is False:
            x = x.permute(0,2,1) #[batch_size, filter_size, height * width]
            x = x.view(bs, self.args.arch.latent_dim, self.args.arch.resolution[0], self.args.arch.resolution[1])
            x = self.enc(x) #(bs, dim, num_cell, num_cell)

        h_enc, c_enc = self.rnn_enc.init_hidden(bs)
        h_dec, c_dec = self.rnn_dec.init_hidden(bs)

        p_global_mean_list = []
        p_global_std_list = []
        q_global_mean_list = []
        q_global_std_list = []
        z_global_list = []

        dec_step = self.dec_step_0.expand(bs, -1, -1, -1)

        for i in range(self.args.arch.draw_step):

            p_global_mean_step, p_global_std_step = \
                self.p_global_net(h_dec.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)
            p_global_std_step = F.softplus(p_global_std_step)

            if phase_generation or (generation_from_step is not None and i >= generation_from_step):

                q_global_mean_step = x.new_empty(bs, self.args.z.z_global_dim)
                q_global_std_step = x.new_empty(bs, self.args.z.z_global_dim)

                if z_global_predefine is None or z_global_predefine.size(1) <= i:
                    z_global_step = Normal(p_global_mean_step, p_global_std_step).rsample()
                else:
                    z_global_step = z_global_predefine.view(bs, -1, self.args.z.z_global_dim)[:, i]

            else:

                if i == 0:
                    rnn_encoder_inp = torch.cat([x, x, h_dec], dim=1)
                else:
                    rnn_encoder_inp = torch.cat([x, x - dec_step, h_dec], dim=1)

                h_enc, c_enc = self.rnn_enc(rnn_encoder_inp, [h_enc, c_enc])

                q_global_mean_step, q_global_std_step = \
                    self.q_global_net(h_enc.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)

                q_global_std_step = F.softplus(q_global_std_step)
                z_global_step = Normal(q_global_mean_step, q_global_std_step).rsample()

            rnn_decoder_inp = self.p_global_decoder_net(z_global_step). \
                reshape(bs, -1, self.args.arch.num_cell, self.args.arch.num_cell)

            rnn_decoder_inp = torch.cat([rnn_decoder_inp, dec_step], dim=1)

            h_dec, c_dec = self.rnn_dec(rnn_decoder_inp, [h_dec, c_dec])

            dec_step = dec_step + self.hid_to_dec_net(h_dec)

            # (bs, dim)
            p_global_mean_list.append(p_global_mean_step)
            p_global_std_list.append(p_global_std_step)
            q_global_mean_list.append(q_global_mean_step)
            q_global_std_list.append(q_global_std_step)
            z_global_list.append(z_global_step)

        global_dec = dec_step

        # (bs, steps, dim, 1, 1)
        p_global_mean_all = torch.stack(p_global_mean_list, 1)[:, :, :, None, None]
        p_global_std_all = torch.stack(p_global_std_list, 1)[:, :, :, None, None]
        q_global_mean_all = torch.stack(q_global_mean_list, 1)[:, :, :, None, None]
        q_global_std_all = torch.stack(q_global_std_list, 1)[:, :, :, None, None]
        z_global_all = torch.stack(z_global_list, 1)[:, :, :, None, None]

        global_dec = self.dec(global_dec) #B*64*128*128
        global_dec = torch.flatten(global_dec, start_dim=2, end_dim=3)#  [batch_size, filter_size, height*width]
        global_dec = global_dec.permute(0, 2, 1)

        pa = [global_dec]
        lv = [z_global_all]
        ss = [p_global_mean_all, p_global_std_all, q_global_mean_all, q_global_std_all]

        return pa, lv, ss

