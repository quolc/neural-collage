# demo of feature blending (+ class-translation): you need .npy files containing target and reference latent variables
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
    parser.add_argument('--z1', type=str, default="z1.npy")
    parser.add_argument('--z2', type=str, default="z2.npy")
    parser.add_argument('--class_mask', type=str, default=None)
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

    z1 = xp.array(np.load(args.z1), dtype=xp.float32)
    z1 = xp.broadcast_to(z1, (args.columns, 128))
    z2 = xp.array(np.load(args.z2), dtype=xp.float32)
    z2 = xp.broadcast_to(z2, (args.columns, 128))

    classes = None

    for i_row in range(args.rows):
        if args.class_mask is not None:
            if classes is None:
                classes = tuple(args.classes) if args.classes is not None and len(args.classes) == 2\
                                              else [np.random.randint(gen.n_classes),
                                                    np.random.randint(gen.n_classes)]

            img_mask = Image.open(args.class_mask).convert("L")

            sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
            ws = []
            for i_size, size in enumerate(sizes):
                resized_mask = xp.array(img_mask.resize((size, size)), dtype=xp.float32) / 255
                w = xp.zeros((args.columns, size, size, gen.n_classes), dtype=xp.float32)
                for i in range(args.columns):
                    weight = i / (args.columns - 1.0)
                    if i_size <= 0:
                        weight *= 0.0
                    w[i, :, :, classes[0]] = 1.0 - resized_mask * weight
                    w[i, :, :, classes[1]] += resized_mask * weight
                ws.append(chainer.Variable(w))
        else:
            if classes is None:
                classes = tuple(args.classes) if args.classes is not None and len(args.classes) == 5\
                                              else [np.random.randint(gen.n_classes),
                                                    np.random.randint(gen.n_classes),
                                                    np.random.randint(gen.n_classes),
                                                    np.random.randint(gen.n_classes),
                                                    np.random.randint(gen.n_classes)]

            sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
            ws = [
                chainer.Variable(xp.zeros((args.columns, size, size, gen.n_classes), dtype=xp.float32)) for size in sizes
            ]
            for i, size in enumerate(sizes):
                ws[i].data[:, :, :, classes[0]] = 1.0
                for j in range(args.columns):
                    ws[i].data[j, :, :, classes[0]] = 1.0 - j / (args.columns - 1.0)
                    ws[i].data[j, :size/2, :size/2, classes[1]] = j / (args.columns - 1.0)
                    ws[i].data[j, :size/2, size/2:, classes[2]] = j / (args.columns - 1.0)
                    ws[i].data[j, size/2:, :size/2, classes[3]] = j / (args.columns - 1.0)
                    ws[i].data[j, size/2:, size/2:, classes[4]] = j / (args.columns - 1.0)

        print(classes)

        # dimension of the feature map for each layer
        sizes_blend = [4, 8, 16, 32, 64, 128, 256]

        # specifying feature blending weights:
        # blends[LAYER_ID][DATA_ID, Y, X, IMAGE_ID]
        # LATENT_ID: 0 = source, 1 = 1st reference, ..., -1 = image to be generated
        blends = [xp.zeros((args.rows, size, size, 3), dtype=xp.float32) for size in sizes_blend]

        # example: applying feature blending to the center region of the 0-th feature map (4x4 resolution)
        blends[0][:, :, :, 0] = 1.0
        blends[0][:, 1:3, 1:3, 0] = 1.0 - i_row / (args.rows - 1.0)  # source weight:    1, 0.75, 0.5, ...
        blends[0][:, 1:3, 1:3, 1] = i_row / (args.rows - 1.0)        # reference weight: 0, 0.25, 0.5, ...

        # you can change the values of blends[LAYER_ID > 0][VERTICAL_RANGE, HORIZONTAL_RANGE, IMAGE_ID]
        # to apply feature blending to later layers
        # e.g., blends[1][2:6, 2:6, 1] = i_row / (args.rows - 1.0)
        # adds more detailed features of the reference image to the center region of the generated image

        # Note: feature blending will not be applied to the i-th layer if blends[i][:, :, -1] == 1.0
        for i in range(len(blends)):
            blends[i][:, :, :, -1] = 1.0
            blends[i][:, :, :, -1] -= blends[i][:, :, :, 0]
            blends[i][:, :, :, -1] -= blends[i][:, :, :, 1]

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen.spatial_interpolation(zs=[z1, z2], weights=ws, blends=blends)

        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        imgs.append(x)
    img = np.stack(imgs)
    _, _, _, h, w = img.shape
    img = img.transpose(0, 3, 1, 4, 2)   # class interpolation on horizontal axis, feature blending on vertiacl axis.
    img = img.reshape((args.rows * h, args.columns * w, 3))

    save_path = os.path.join(out, 'interpolated_images.png')
    if not os.path.exists(out):
        os.makedirs(out)
    Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    main()
