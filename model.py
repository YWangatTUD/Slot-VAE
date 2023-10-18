import torch
from torch import nn
from module import ImgEncoder, LocalLatentEncoder, LocalSampler, StructDRAW, Decoder, GlobalToSlots, GlobalEncoder, GlobalToFeature
from torch.distributions import Normal, kl_divergence
from typing import List, Tuple


class SLOT_VAE(nn.Module):

    def __init__(self, args):
        super(SLOT_VAE, self).__init__()
        self.args = args

        self.img_encoder = ImgEncoder(self.args)
        self.global_struct_draw = StructDRAW(self.args)
        self.p_z_given_x = LocalLatentEncoder(self.args)
        self.p_z_given_g = self.p_z_given_x

        self.local_latent_sampler = LocalSampler(self.args)
        self.decoder = Decoder(self.args)


    def forward(self, x: torch.Tensor) -> Tuple:
        bs = x.size(0)
        device = x.get_device()
        p_zs = Normal(torch.zeros(bs, self.args.arch.num_slots, self.args.arch.latent_dim).to(device),\
                      torch.ones(bs, self.args.arch.num_slots, self.args.arch.latent_dim).to(device))

        img_enc = self.get_img_enc(x)

        f_zg, zg_given_x, ss = self.global_struct_draw(img_enc)
        p_zg_mean_all, p_zg_std_all, q_zg_mean_all, q_zg_std_all = ss
        p_zg_all = Normal(p_zg_mean_all, p_zg_std_all)
        q_zg_given_x_all = Normal(q_zg_mean_all, q_zg_std_all)

        q_zs_given_x_mean, q_zs_given_x_std, slots_init = self.ss_q_z_given_x(img_enc, None)
        q_zs_given_x = Normal(q_zs_given_x_mean, q_zs_given_x_std)
        zs_given_x = q_zs_given_x.rsample()

        p_zs_given_g_mean, p_zs_given_g_std, slots_init = self.ss_p_z_given_g(f_zg[0], slots_init)
        p_zs_given_g = Normal(p_zs_given_g_mean, p_zs_given_g_std)

        recon = self.lv_p_x_given_z(zs_given_x)

        p_dists = [p_zg_all, p_zs_given_g, p_zs]

        q_dists = [q_zg_given_x_all, q_zs_given_x]

        log_like, kl, log_imp = \
            self.elbo(x, p_dists, q_dists, zs_given_x, zg_given_x, recon)

        self.log = {}

        if self.args.log.phase_log:
            img_recon_from_q_g = self.get_recon_from_q_g(f_zg, slots_init)
            self.log = {
                'zs': zs_given_x,
                'p_zs_given_g_std': p_zs_given_g_std,
                'p_zs_given_g_mean': p_zs_given_g_mean,

                'q_zs_given_x_std': q_zs_given_x_std,
                'q_zs_given_x_mean': q_zs_given_x_mean,

                'recon': recon[0],
                'recon_from_q_g': img_recon_from_q_g[0],
                'recon_from_q_g_slots': img_recon_from_q_g[1],
                'recon_from_q_g_masks': img_recon_from_q_g[2],
                'log_prob_x_given_g': Normal(img_recon_from_q_g[0], self.args.const.likelihood_sigma).
                    log_prob(x).flatten(start_dim=1).sum(1),
            }

        ss = [p_zs_given_g_mean, p_zs_given_g_std, q_zg_mean_all, q_zg_std_all]

        return recon, log_like, kl, log_imp, zs_given_x, zg_given_x, ss, self.log

    def get_recon_from_q_g(
            self,
            f_zg,
            slots_init,
            phase_use_mode: bool = False
    ) -> Tuple:
        p_zs_given_g_mean, p_zs_given_g_std, slots_init = self.ss_p_z_given_g(f_zg[0], slots_init)
        p_zs_given_g = Normal(p_zs_given_g_mean, p_zs_given_g_std)
        zs_given_g = p_zs_given_g.rsample()

        recon = self.lv_p_x_given_z(zs_given_g, phase_use_mode=phase_use_mode)

        return recon

    def sample(self, phase_use_mode: bool = False):

        dummy_x = torch.zeros([self.args.arch.num_samples, 1, 1]).to('cuda:0')
        f_zg, zg_given_x, ss = self.global_struct_draw(dummy_x, phase_generation=True)
        p_zs_given_g_mean, p_zs_given_g_std, slots_init = self.ss_p_z_given_g(f_zg[0], None)
        p_zs_given_g = Normal(p_zs_given_g_mean, p_zs_given_g_std)
        zs_given_g = p_zs_given_g.rsample()

        recon = self.lv_p_x_given_z(zs_given_g, phase_use_mode=phase_use_mode)

        return recon

    def elbo(self,
             x: torch.Tensor,
             p_dists: List,
             q_dists: List,
             zs_given_x,
             zg_given_x,
             recon) -> Tuple:

        bs = x.size(0)
        zg = zg_given_x[0]

        p_zg_all, p_zs_given_g, p_zs = p_dists

        q_zg_given_x_all, q_zs_given_x = q_dists

        img_recon, recons, masks = recon

        aux_kl_zs = kl_divergence(q_zs_given_x, p_zs)

        kl_zs = kl_divergence(q_zs_given_x, p_zs_given_g)

        kl_zg = kl_divergence(q_zg_given_x_all, p_zg_all)

        log_like = Normal(img_recon, self.args.const.likelihood_sigma).log_prob(x)

        log_imp_list = []
        if self.args.log.phase_nll:

            log_imp_zs = p_zs_given_g.log_prob(zs_given_x) - \
                           q_zs_given_x.log_prob(zs_given_x)

            log_imp_zg = p_zg_all.log_prob(zg) - q_zg_given_x_all.log_prob(zg)

            log_imp_list = [
                log_imp_zs.view(bs, self.args.arch.num_slots, self.args.arch.slot_size).flatten(start_dim=1).sum(1),
                log_imp_zg.flatten(start_dim=1).sum(1),
            ]

        return log_like.flatten(start_dim=1).sum(1), \
               [
                   aux_kl_zs.view(bs, self.args.arch.num_slots, self.args.arch.slot_size).flatten(start_dim=1).sum(
                       -1),
                   kl_zs.view(bs, self.args.arch.num_slots, self.args.arch.slot_size).flatten(start_dim=1).sum(-1),
                   kl_zg.flatten(start_dim=2).sum(-1),
               ], log_imp_list

    def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: (bs, inp_channel, img_h, img_w)
        :return: img_enc: (bs, dim, num_cell, num_cell)
        """

        img_enc = self.img_encoder(x)

        return img_enc

    def ss_p_z_given_g(self, global_dec: torch.Tensor, slot_init) -> List:
        """

        :param x: sample of z_global variable (bs, dim, 1, 1)
        :return:
        """
        p_zs_given_zg_mean, p_zs_given_zg_std, slots_init = self.p_z_given_g(global_dec, slot_init)

        return p_zs_given_zg_mean, p_zs_given_zg_std, slots_init

    def ss_q_z_given_x(self, img_enc: torch.Tensor, slot_init) -> List:
        """

        :param x: sample of z_global variable (bs, dim, 1, 1)
        :return:
        """
        q_zs_given_x_mean, q_zs_given_x_std, slots_init = self.p_z_given_x(img_enc, slot_init)

        return q_zs_given_x_mean, q_zs_given_x_std, slots_init

    def lv_p_x_given_z(self, zs,  phase_use_mode: bool = False) -> Tuple:
        """
        :param z: (bs, z_what_dim)
        :return:
        """
        batch_size = zs.shape[0]
        zs = zs.view(zs.shape[0] * self.args.arch.num_slots, self.args.arch.slot_size, 1, 1)
        decoder_in = zs.repeat(1, 1, self.args.arch.decoder_resolution[0], self.args.arch.decoder_resolution[1])

        img_recon, recons, masks = self.decoder(decoder_in, batch_size, self.args.arch.num_slots, self.args.arch.in_channels, self.args.arch.resolution[0], self.args.arch.resolution[1])

        recon = [img_recon, recons, masks]

        return recon