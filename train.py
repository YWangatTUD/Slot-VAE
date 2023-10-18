import os
import time
import torch
import math
import yaml

import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from data import Blender
from utils import save_ckpt, load_ckpt, print_schedule, \
    visualize, linear_schedule, log_mean_exp, to_rgb_from_tensor

from config import get_config

from model import SLOT_VAE
import wandb
import os



def main():

    torch.backends.cudnn.benchmark = True

    args = get_config()[0]

    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    np.random.seed(args.train.seed)

    model_dir = os.path.join(args.model_dir, args.exp_name)
    summary_dir = os.path.join(args.summary_dir, args.exp_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # torch.manual_seed(args.seed)
    args.train.num_gpu = torch.cuda.device_count()
    with open(os.path.join(summary_dir, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)
    if args.data.dataset == 'blender':
        train_data = Blender(args, mode='train')
    else:
        raise NotImplemented

    train_loader = DataLoader(
        train_data, batch_size=args.train.batch_size, shuffle=True, drop_last=True, num_workers=6)
    num_train = len(train_data)

    model = SLOT_VAE(args)
    model.to(device)
    num_gpu = 1
    '''args.device_ids = [0, 1, 6, 7]
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        num_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)'''
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)

    warmup_steps_pct = args.train.warmup_steps_pct
    decay_steps_pct = args.train.decay_steps_pct
    total_steps = args.train.epoch * num_train

    def warm_and_decay_lr_scheduler(step: int):
        warmup_steps = warmup_steps_pct * total_steps
        decay_steps = decay_steps_pct * total_steps
        assert step < total_steps
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = 1
        factor *= args.train.scheduler_gamma ** (step / decay_steps)
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)


    global_step = 0

    '''if args.last_ckpt:
        global_step, args.train.start_epoch = \
            load_ckpt(model, optimizer, args.last_ckpt, device)'''

    args.train.global_step = global_step
    args.log.phase_log = False

    end_time = time.time()
    wandb.init(project='SlotVAE', name='arrow_lr1e-4')
    for epoch in range(int(args.train.start_epoch), args.train.epoch):

        local_count = 0
        last_count = 0
        for batch_idx, sample in enumerate(train_loader):

            imgs = sample.to(device)

            hyperparam_anneal(args, global_step)

            global_step += 1

            phase_log = global_step % args.log.print_step_freq == 0 or global_step == 1
            args.train.global_step = global_step
            args.log.phase_log = phase_log

            recon, log_like, kl, _, _, _, _, log = \
                model(imgs)

            aux_kl_zs, kl_zs, kl_zg = kl

            aux_kl_zs_raw = aux_kl_zs.mean(dim=0)
            kl_zs_raw = kl_zs.mean(dim=0)
            kl_zg_raw = kl_zg.sum(dim=-1).mean(dim=0)
            log_like = log_like.mean(dim=0)

            aux_kl_zs = aux_kl_zs_raw * args.train.beta_aux_zs
            kl_zs = kl_zs_raw * args.train.beta_zs
            kl_zg = kl_zg_raw * args.train.beta_zg

            total_loss = - (log_like - kl_zs - kl_zg - aux_kl_zs)

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(model.parameters(), args.train.cp)
            optimizer.step()
            scheduler.step()

            local_count += imgs.data.shape[0]
            if phase_log:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_schedule(global_step, epoch, local_count, count_inter,
                               num_train, total_loss, time_inter)
                end_time = time.time()

                ############### Show reconstructed image from zs and slot components
                out = torch.cat(
                        [
                            imgs.cpu().detach()[:args.log.num_summary_img].view(-1, args.data.inp_channel,
                                                                                args.data.img_h,
                                                                                args.data.img_w).unsqueeze(1),  # original images
                            recon[0].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                                view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w).unsqueeze(1),  # reconstructions
                            recon[1].cpu().detach()[:args.log.num_summary_img].clamp(0, 1) * recon[2].cpu().detach()[:args.log.num_summary_img] + (1 - recon[2].cpu().detach()[:args.log.num_summary_img]),  # each slot
                        ],
                        dim=1,
                    )

                batch_size, num_slots, C, H, W = recon[1].shape
                images = make_grid(
                    out.view(args.log.num_summary_img * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1], pad_value=1
                )
                image = wandb.Image(images, caption="reconstruction_overall_with_slots")
                wandb.log({"reconstruction_overall_with_slots": image}, global_step)

                ############### Show reconstructed image from zg_zs' and slot components
                out = torch.cat(
                        [  imgs.cpu().detach()[:args.log.num_summary_img].view(-1, args.data.inp_channel,
                                                                                args.data.img_h,
                                                                                args.data.img_w).unsqueeze(1),
                            log['recon_from_q_g'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                                view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w).unsqueeze(1),
                            # reconstructions
                            log['recon_from_q_g_slots'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1) * log['recon_from_q_g_masks'].cpu().detach()[:args.log.num_summary_img] + (1 - log['recon_from_q_g_masks'].cpu().detach()[:args.log.num_summary_img]),  # each slot
                        ],
                        dim=1,
                    )
                batch_size, num_slots, C, H, W = recon[1].shape
                images = make_grid(
                    out.view(args.log.num_summary_img * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1], pad_value=1
                )
                image = wandb.Image(images, caption="reconstruction_from_q_g_with_slots")
                wandb.log({"reconstruction_from_q_g_with_slots": image}, global_step)

                elbo = (log_like.item() - kl_zs_raw.item() - kl_zg_raw.item())
                wandb.log(
                    {'train/total_loss': total_loss.item(), 'train/log_like': log_like.item(),
                     'train/zs_KL': kl_zs.item(),
                     'train/zg_KL': kl_zg.item(), 'train/aux_zs_KL': aux_kl_zs.item(),
                     'train/log_prob_x_given_g': log['log_prob_x_given_g'].mean(0).item(),
                     'train/lr': optimizer.param_groups[0]['lr'], 'train/elbo': elbo}, global_step
                )

                ######################################## generation ########################################

                with torch.no_grad():
                    model.eval()
                    if num_gpu > 1:
                        sample = model.module.sample()[0]
                    else:
                        sample = model.sample()[0]
                    model.train()

                grid_image = make_grid(
                    sample[0].cpu().detach().clamp(0, 1),
                    args.log.num_img_per_row, normalize=False, pad_value=1)
                image = wandb.Image(grid_image, caption="generation")
                wandb.log({"generation": image}, global_step)

                ###################################### generation end ######################################

                last_count = local_count

        if epoch % args.log.save_epoch_freq == 0 and epoch != 0:
            save_ckpt(model_dir, model, optimizer, global_step, epoch,
                      local_count, args.train.batch_size, num_train)

    save_ckpt(model_dir, model, optimizer, global_step, epoch,
              local_count, args.train.batch_size, num_train)


def hyperparam_anneal(args, global_step):
    if args.train.beta_aux_zs_anneal_end_step == 0:
        args.train.beta_aux_zs = args.train.beta_aux_zs_anneal_start_value
    else:
        args.train.beta_aux_zs = linear_schedule(
            global_step,
            args.train.beta_aux_zs_anneal_start_step,
            args.train.beta_aux_zs_anneal_end_step,
            args.train.beta_aux_zs_anneal_start_value,
            args.train.beta_aux_zs_anneal_end_value
        )

    ########################### split here ###########################
    if args.train.beta_zs_anneal_end_step == 0:
        args.train.beta_zs = args.train.beta_zs_anneal_start_value
    else:
        args.train.beta_zs = linear_schedule(
            global_step,
            args.train.beta_zs_anneal_start_step,
            args.train.beta_zs_anneal_end_step,
            args.train.beta_zs_anneal_start_value,
            args.train.beta_zs_anneal_end_value
        )

    if args.train.beta_zg_anneal_end_step == 0:
        args.train.beta_zg = args.train.beta_zg_anneal_start_value
    else:
        args.train.beta_zg = linear_schedule(
            global_step,
            args.train.beta_zg_anneal_start_step,
            args.train.beta_zg_anneal_end_step,
            args.train.beta_zg_anneal_start_value,
            args.train.beta_zg_anneal_end_value
        )
    return


if __name__ == '__main__':
    main()
