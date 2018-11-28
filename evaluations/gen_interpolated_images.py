# non-spatial class interpolation
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
    parser.add_argument('--n_intp', type=int, default=5)
    parser.add_argument('--n_zs', type=int, default=5)
    parser.add_argument('--classes', type=int, nargs="*", default=None)
    parser.add_argument('--seed', type=int, default=1236)
    args = parser.parse_args()

    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    gen = load_models(config)
    gen.to_gpu()
    chainer.serializers.load_npz(args.snapshot, gen)

    out = args.results_dir

    xp = gen.xp
    imgs = []
    classes = tuple(args.classes) if args.classes is not None else [np.random.randint(config.models['generator']['args']['n_classes']),
                                                                    np.random.randint(config.models['generator']['args']['n_classes'])]
    for _ in range(args.n_zs):
        z = xp.array([np.random.normal(size=(128,))] * args.n_intp, xp.float32)
        ys = xp.array([[classes[0], classes[1]]] * args.n_intp, dtype=xp.int32)
        ws_y = xp.array([np.linspace(0, 1, args.n_intp)[::-1], np.linspace(0, 1, args.n_intp)], dtype=xp.float32).T
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z=z, y=ys, weights=ws_y)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        imgs.append(x)
    img = np.stack(imgs)
    _, _, _, h, w = img.shape
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape((args.n_zs * h, args.n_intp * w, 3))

    save_path = os.path.join(out, 'interpolated_images_{}-{}.png'.format(classes[0], classes[1]))
    if not os.path.exists(out):
        os.makedirs(out)
    Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    main()
