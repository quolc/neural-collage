# spatial class translation demo (we strongly recommend to try web-based demo for interactive morphing)
import os, sys
import numpy as np
import argparse
import chainer
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
import yaml
import source.yaml_utils as yaml_utils


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--columns', type=int, default=5)
    parser.add_argument('--classes', type=int, nargs="*", default=None)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    np.random.seed(args.seed)

    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    gen = load_models(config)
    gen.to_gpu()
    chainer.serializers.load_npz(args.snapshot, gen)

    out = args.results_dir

    xp = gen.xp
    imgs = []

    for _ in range(args.rows):
        z = xp.array([np.random.normal(size=(128,)) for _ in range(args.columns)], xp.float32)
        classes = tuple(args.classes) if args.classes is not None and len(args.classes) == 4\
                                      else [np.random.randint(gen.n_classes),
                                            np.random.randint(gen.n_classes),
                                            np.random.randint(gen.n_classes),
                                            np.random.randint(gen.n_classes)]

        sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
        # zero-weights
        ws = [
            chainer.Variable(xp.zeros((args.columns, size, size, gen.n_classes), dtype=xp.float32)) for size in sizes
        ]
        for i, size in enumerate(sizes):
            ws[i].data[:, :size/2, :size/2, classes[0]] = 1.0
            ws[i].data[:, :size/2, size/2:, classes[1]] = 1.0
            ws[i].data[:, size/2:, :size/2, classes[2]] = 1.0
            ws[i].data[:, size/2:, size/2:, classes[3]] = 1.0

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen.spatial_interpolation(z, ws)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        imgs.append(x)
    img = np.stack(imgs)
    _, _, _, h, w = img.shape
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape((args.rows * h, args.columns * w, 3))

    save_path = os.path.join(out, 'interpolated_images_{}-{}-{}-{}.png'.format(classes[0], classes[1],
                                                                               classes[2], classes[3]))
    if not os.path.exists(out):
        os.makedirs(out)
    Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    main()
