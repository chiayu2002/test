import argparse
import os
from os import path
import time
import copy
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Trainer, Evaluator
from graf.config import get_data, build_models, build_lr_scheduler, build_optimizers, load_config
from graf.utils import count_trainable_parameters, get_nsamples, save_images, get_zdist
from graf.transforms import ImgToPatch

# from GAN_stability.gan_training import utils
from GAN_stability.gan_training.train_mod import update_average

import wandb


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('--config', default='/Data/home/vicky/test/configs/default.yaml', type=str, help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
        
    # Short hands
    batch_size = config['training']['batch_size']
    restart_every = config['training']['restart_every']
    fid_every = config['training']['fid_every']
    save_every = config['training']['save_every']
    backup_every = config['training']['backup_every']
    save_best = config['training']['save_best']
    assert save_best=='fid' or save_best=='kid', 'Invalid save best metric!'

    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr)
    print(f"Dataset size: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Please check your data path and files.")
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True, 
        generator=torch.Generator(device='cuda:0')
    )

    with open('label_value0110.txt', 'w') as f:
        for file_path, label in train_loader.dataset.labels.items():
            f.write(f"文件路徑: {file_path}, label: {label}\n")
    
    # Create models
    generator, discriminator = build_models(config)
    print('Generator params: %d' % count_trainable_parameters(generator))
    print('Discriminator params: %d, channels: %d' % (count_trainable_parameters(discriminator), discriminator.nc))
    print(generator.render_kwargs_train['network_fn'])
    print(discriminator)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optimizer, d_optimizer = build_optimizers(
        generator, discriminator, config
    )

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    config.update({
        "g_optimizer": g_optimizer.__class__.__name__,
        "d_optimizer": d_optimizer.__class__.__name__,
    })

    wandb.init(project="graftest", entity="vicky20020808", allow_val_change=True, config=config)


    model_checkpoint = "model_checkpoint.pth"
    checkpoint_data = {
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
        **{f"generator_{key}": val.state_dict() for key, val in generator.module_dict.items()}  # 特別處理 generator
    }

    torch.save(checkpoint_data, model_checkpoint)
    wandb.save(model_checkpoint)

    # 上傳 checkpoint 到 W&B
    artifact = wandb.Artifact(name="model_checkpoint", type="checkpoint")
    artifact.add_file(model_checkpoint)
    wandb.log_artifact(artifact)

    # Get model file
    model_file = config['training']['model_file']
    stats_file = 'stats.p'

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Save for tests
    ntest = batch_size
    # x_real, x_label = get_nsamples(train_loader, ntest)
    #ytest = torch.zeros(ntest)
    ztest = zdist.sample((ntest,))
    ptest = torch.stack([generator.sample_pose() for i in range(ntest)])
    ptest_list = []
    label_list = []
    label_vedio = []
    # third_value_map = {0: 359, 0.25: 89, 0.5: 179, 0.75: 269}
    for i in range(ntest):
        if i < 4:
            # y_value = 0.4
            first_value = 0
        else:
            # y_value = 0.3
            first_value = 1
        # u_value = 0.25*(i%4)
        # third_value = third_value_map[u_value]
        # ptest_list.append(generator.sample_test_pose(u_value, y_value)) #generate pose
        label_list.append([first_value])
    # ptest = torch.stack(ptest_list)
    print(f"posetest:{ptest}")
    label_test = torch.tensor(label_list)
    print(f"labeltest:{label_test}")

    # u_map = {0: 359, 0.125: 44, 0.25: 89, 0.375: 134, 0.5: 179, 0.625: 224, 0.75: 269, 0.875: 314}
    for i in range(ntest):
        # u_value = 0.125*(i%8)
        # third_value = u_map[u_value]
        label_vedio.append([0])

    # save_images(x_real, path.join(out_dir, 'real.png'))

    # Test generator
    if config['training']['take_model_average']:
        generator_test = copy.deepcopy(generator)
        # we have to change the pointers of the parameter function in nerf manually
        generator_test.parameters = lambda: generator_test._parameters
        generator_test.named_parameters = lambda: generator_test._named_parameters
        # checkpoint_io.register_modules(**{k+'_test': v for k, v in generator_test.module_dict.items()})
    else:
        generator_test = generator

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator_test, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)

    # Initialize fid+kid evaluator
    if fid_every > 0:
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(train_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)

    # Train
    tstart = t0 = time.time()

    # # Load checkpoint if it exists
    try:
        load_dict = torch.load(model_file)
    except FileNotFoundError:
        it = epoch_idx = -1
        fid_best = float('inf')
        kid_best = float('inf')
    else:
        it = load_dict.get('it', -1)
        epoch_idx = load_dict.get('epoch_idx', -1)
        fid_best = load_dict.get('fid_best', float('inf'))
        kid_best = load_dict.get('kid_best', float('inf'))
        torch.load_stats(stats_file)

    # Reinitialize model average if needed
    if (config['training']['take_model_average']
      and config['training']['model_average_reinit']):
        update_average(generator_test, generator, 0.)

    # Learning rate anneling
    d_lr = d_optimizer.param_groups[0]['lr']
    g_lr = g_optimizer.param_groups[0]['lr']
    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)  #控制優化器的學習率
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)
    # ensure lr is not decreased again
    d_optimizer.param_groups[0]['lr'] = d_lr
    g_optimizer.param_groups[0]['lr'] = g_lr

    # Trainer
    trainer = Trainer(
        generator, discriminator, g_optimizer, d_optimizer,
        use_amp=config['training']['use_amp'])

    print('it {}: start with LR:\n\td_lr: {}\tg_lr: {}'.format(it, d_optimizer.param_groups[0]['lr'], g_optimizer.param_groups[0]['lr']))

    wandb.watch(discriminator, log="all")
    print(generator.module_dict)
    wandb.watch(generator.module_dict['generator'], log="all")

    def save_best_model(metric, metric_best, metric_name, model_name):
        if metric < metric_best:
            metric_best = metric
            print(f'Saving best model based on {metric_name}...')
            wandb.save(model_name)
            wandb.log({
                "iteration": it,
                "epoch_idx": epoch_idx,
                f"{metric_name}_best": metric_best,
                "fid_best": fid_best,
                "kid_best": kid_best
            })
            torch.cuda.empty_cache()
        return metric_best

    # Training loop
    print('Start training...')
    while True:
        epoch_idx += 1
        print('Start epoch %d...' % epoch_idx)

        for x_real, real_label in train_loader:
            t_it = time.time()
            it += 1
            generator.ray_sampler.iterations = it   # for scale annealing

            # Sample patches for real data
            rgbs = img_to_patch(x_real.to(device))          # N_samples x C

            # Discriminator updates
            z = zdist.sample((batch_size,)) #torch.Size([8, 256])
            dloss, reg = trainer.discriminator_trainstep(rgbs, real_label, z=z)

            wandb.log({
            "loss/discriminator": dloss,
            "loss/regularizer": reg,
            "iteration": it
            })

            # Generators updates
            if config['nerf']['decrease_noise']:
              generator.decrease_nerf_noise(it)

            z = zdist.sample((batch_size,))
            gloss = trainer.generator_trainstep(real_label, z)

            wandb.log({
            "loss/generator": gloss,
            "iteration": it
            })

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])

            # Update learning rate
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']

            wandb.log({
            "Generator LR": g_lr,
            "Discriminator LR": d_lr
            })

            dt = time.time() - t_it
            # Print stats
            if ((it + 1) % config['training']['print_every']) == 0:
                print('[%s epoch %0d, it %4d, t %0.3f] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                  % (config['expname'], epoch_idx, it + 1, dt, dloss, gloss, reg))

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
            # if it >1:
                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test, poses=ptest)
                wandb.log({
                "sample/rgb": [wandb.Image(rgb)],
                "sample/depth": [wandb.Image(depth)],
                "sample/acc": [wandb.Image(acc)],
                "iteration": it
                })

            # (v) Compute fid if necessary
            if fid_every > 0 and ((it + 1) % fid_every) == 0:
                fid, kid = evaluator.compute_fid_kid(real_label)
                wandb.log({
                 "validation/fid": fid,
                    "validation/kid": kid,
                    "iteration": it
                })
                torch.cuda.empty_cache()
                if save_best == 'fid':
                    fid_best = save_best_model(fid, fid_best, "fid", 'model_best.pth')
                elif save_best == 'kid':
                    kid_best = save_best_model(kid, kid_best, "kid", 'model_best.pth')
                    generator.save(f'0108_for_graf_{it + 1}epochs.h5')

            # (vi) Create video if necessary
            if ((it+1) % config['training']['video_every']) == 0:
                N_samples = 4
                zvid = zdist.sample((N_samples,))
                basename = os.path.join(out_dir, '{}_{:06d}_'.format(os.path.basename(config['expname']), it))
                evaluator.make_video(basename, zvid, label_vedio, ptest, as_gif=False)

            # (i) Backup if necessary
            if ((it + 1) % backup_every) == 0:
                print('Saving backup...')
                wandb.save(f'model_{it:08d}.pth')
                wandb.log({
                    "iteration": it,
                    "epoch_idx": epoch_idx,
                    "fid_best": fid_best,
                    "kid_best": kid_best
                    })

            # (vi) Save checkpoint if necessary
            if time.time() - t0 > save_every:
                print('Saving checkpoint...')
                wandb.save(f'model_checkpoint_{it}.pth')
                wandb.log({
                    "iteration": it,
                    "epoch_idx": epoch_idx,
                    "fid_best": fid_best,
                    "kid_best": kid_best
                    })
                t0 = time.time()

                if (restart_every > 0 and t0 - tstart > restart_every):
                    exit(3)