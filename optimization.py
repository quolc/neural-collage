import sys, os
import argparse
import yaml
import chainer
from chainer import Variable
from chainer import optimizers
import chainer.functions as F

sys.path.append(os.getcwd())
import source.yaml_utils as yaml_utils

import numpy as np
from PIL import Image


def reconstruction_loss(dis, recon, gt):
    with chainer.using_config('train', False):
        v1 = dis.feature_vector(recon)
        v2 = dis.feature_vector(gt)
    denom = F.sqrt(F.batch_l2_norm_squared(v1) * F.batch_l2_norm_squared(v2))

    xp = dis.xp
    sum = Variable(xp.array(0.0, dtype=xp.float32))
    for i in range(gt.shape[0]):
        sum += F.matmul(v1[i], v2[i], transb=True) / denom[i]
    cos_dist2 = - sum
    return cos_dist2


def pixel_loss(recon, gt):
    return F.mean_squared_error(recon, gt)


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    if 'encoder' in config.models:
        enc_conf = config.models['encoder']
        enc = yaml_utils.load_model(enc_conf['fn'], enc_conf['name'], enc_conf['args'])
        return gen, dis, enc
    else:
        return gen, dis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='path to the encoder .npz file')
    parser.add_argument('--gen_model', type=str, default='',
                        help='path to the generator .npz file')
    parser.add_argument('--dis_model', type=str, default='',
                        help='path to the discriminator .npz file')
    parser.add_argument('--src_class', type=int, help="target class")
    parser.add_argument('--input', type=str, help="input image")
    parser.add_argument('--mode', type=str, default='noaux', help='set "aux" if you want to use auxiliary network')
    parser.add_argument('--noenc', action='store_true', help='specify if you do not want to use encoder')
    parser.add_argument('--iter_opt', type=int, default=200)
    parser.add_argument('--result_dir', type=str, default="opt_output")
    args = parser.parse_args()

    np.random.seed(1234)

    config_path = args.config_path
    gpu = args.gpu
    enc_model = args.enc_model
    gen_model = args.gen_model
    dis_model = args.dis_model
    src_class = args.src_class

    optimize_iterations = args.iter_opt
    lmd_pixel = 0.2

    use_aux = args.mode == "aux"
    no_enc = args.noenc

    # optimizer setup
    config = yaml_utils.Config(yaml.load(open(config_path)))
    chainer.cuda.get_device_from_id(gpu).use()
    if 'encoder' in config.models:
        gen, dis, enc = load_models(config)
    else:
        gen, dis = load_models(config)
        enc = None

    chainer.serializers.load_npz(gen_model, gen, strict=False)
    chainer.serializers.load_npz(dis_model, dis)

    gen.to_gpu(device=gpu)
    dis.to_gpu(device=gpu)

    if not no_enc and enc is not None:
        chainer.serializers.load_npz(enc_model, enc)
        enc.to_gpu(device=gpu)

    xp = gen.xp

    # load image
    img_size = config['dataset']['args']['size']
    inp_size = img_size
    if 'input_size' in config['updater']['args']:
        inp_size = config['updater']['args']['input_size']
    patch_inp = np.asarray(Image.open(args.input).resize((inp_size, inp_size)))[:,:,:3]
    patch_gt = np.asarray(Image.open(args.input).resize((img_size, img_size)))[:,:,:3]

    x = patch_inp.astype(np.float32) / 128. - 1
    x = Variable(xp.asarray(x).transpose(2, 0, 1).reshape(1, 3, inp_size, inp_size))
    gt = patch_gt.astype(np.float32) / 128. - 1
    gt = Variable(xp.asarray(gt).transpose(2, 0, 1).reshape(1, 3, img_size, img_size))
    c = Variable(xp.array([src_class], dtype=xp.int32))

    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        if enc is not None and not no_enc:
            z = enc(x, c)
        else:
            z = Variable(xp.array([np.random.normal(size=(128,))], xp.float32))
        if use_aux:
            z2 = gen.forward_A(z)

    if use_aux:
        optimizer = optimizers.Adam(alpha=0.015)
        target = chainer.links.Parameter(z2.data)
    else:
        optimizer = optimizers.Adam(alpha=0.075)
        target = chainer.links.Parameter(z.data)
    optimizer.setup(target)

    # optimization body
    print("optimizing...")
    for iteration in range(optimize_iterations):
        target.cleargrads()
        _target = target.W

        with chainer.using_config('train', False):
            if use_aux:
                recon, _, _, z = gen.forward(zeta=_target, y=c, return_zs=True)
            else:
                recon = gen(z=_target, y=c)

        l1 = reconstruction_loss(dis, recon, gt)
        l2 = lmd_pixel * pixel_loss(recon, gt)
        loss = l1 + l2

        loss.backward()
        target.W.grad = _target.grad
        optimizer.update()

        if not use_aux:
            z = target.W

        print(iteration, l1.data, l2.data, loss.data)

        if iteration % 10 == 0:
            img = recon.data.get()[0].transpose(1,2,0) * 127.5 + 127.5
            patch_recon = np.asarray(np.clip(img, 0, 255), dtype=np.uint8)

            if not os.path.exists(args.result_dir):
                os.mkdir(args.result_dir)
            Image.fromarray(patch_recon).save("{}/opt_{}.png".format(args.result_dir, iteration))
            with open("{}/opt_{}.npy".format(args.result_dir, iteration), "wb") as f:
                np.save(f, z.data.get())
    print("done.")

    img = recon.data.get()[0].transpose(1,2,0) * 127.5 + 127.5
    patch_recon = np.asarray(np.clip(img, 0, 255), dtype=np.uint8)

    # reconstruction preview
    Image.fromarray(patch_recon).save("{}/opt_final.png".format(args.result_dir))
    with open("{}/opt_final.npy".format(args.result_dir), "wb") as f:
        np.save(f, z.data.get())

    # z save
    with open("opt-z.npy", "wb") as f:
        np.save(f, z.data.get())

if __name__ == "__main__":
    main()