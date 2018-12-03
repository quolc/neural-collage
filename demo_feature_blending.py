import flask
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import base64
import os
import secrets
import argparse
import yaml
import chainer
from chainercv.transforms import resize

from PIL import Image
from scipy.ndimage.filters import gaussian_filter

import source.yaml_utils as yaml_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
parser.add_argument('--gen_model', type=str, default='',
                    help='path to the generator .npz file')
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10000000  # allow 10 MB post

config_path = args.config_path
snapshot_path = args.gen_model

config = yaml_utils.Config(yaml.load(open(args.config_path)))


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen

gen = load_models(config)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    gen.to_gpu()
chainer.serializers.load_npz(args.gen_model, gen)

xp = gen.xp


def gen_images(z, c):
    xs = []

    nb = config['batchsize']
    for i in range(0, len(z), nb):
        # use scbn_version generator for uniform class generation
        cs = c[i:i+nb]

        sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
        ws = [
            chainer.Variable(xp.zeros((len(cs), size, size, gen.n_classes), dtype=xp.float32)) for size in sizes
        ]
        for i_size, size in enumerate(sizes):
            for j in range(len(cs)):
                ws[i_size].data[j, :, :, cs[j]] = 1.0

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen.spatial_interpolation(z=z[i:i+nb], weights=ws)

        x = x.data
        if args.gpu >= 0:
            x = x.get()

        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).transpose((0,2,3,1))
        xs.append(x)
    return np.vstack(xs)


def gen_blended_images(z_src, z_ref, c, mask, lmds, interpolation=16):
    z_src = xp.broadcast_to(z_src, (interpolation, 128))
    z_ref = xp.broadcast_to(z_ref, (interpolation, 128))

    sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
    ws = [
        chainer.Variable(xp.zeros((interpolation, size, size, gen.n_classes), dtype=xp.float32)) for size in sizes
    ]
    for i_size, size in enumerate(sizes):
        ws[i_size].data[:, :, :, c[0]] = 1.0

    sizes_blend = [4, 8, 16, 32, 64, 128, 256]
    blends = [xp.zeros((interpolation, size, size, 3), dtype=xp.float32) for size in sizes_blend]

    masks_resized = [xp.array(resize(mask, (size, size))).reshape((size, size)) for size in sizes_blend]

    for i in range(interpolation):
        blends[0][i, :, :, 1] = lmds[0] * i / (interpolation - 1.0) * masks_resized[0]
        blends[0][i, :, :, 0] = 1.0 - blends[0][i, :, :, 1]

    for layer in range(1, len(lmds)):
        for i in range(interpolation):
            blends[layer][i, :, :, 1] = lmds[layer] * i / (interpolation - 1.0) * masks_resized[layer]

    for i in range(len(blends)):
        blends[i][:, :, :, -1] = 1.0
        blends[i][:, :, :, -1] -= blends[i][:, :, :, 0]
        blends[i][:, :, :, -1] -= blends[i][:, :, :, 1]

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        x = gen.spatial_interpolation(zs=[z_src, z_ref], weights=ws, blends=blends)

    x = x.data
    if args.gpu >= 0:
        x = x.get()

    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).transpose((0,2,3,1))
    return x

@app.route('/')
def index():
    return render_template('demo_feature_blending/index.html',
                           dataset_name=config['dataset']['dataset_name'])


@app.route('/generate', methods=['POST'])
def generate():
    z = xp.array(request.json["zs"], dtype=xp.float32)
    c = xp.array(request.json["cs"], dtype=xp.int32)

    generated_images = gen_images(z, c)
    paths = []
    if not os.path.exists("static/demo_feature_blending/generated"):
        os.mkdir("static/demo_feature_blending/generated")
    for i, img in enumerate(generated_images):
        path = "static/demo_feature_blending/generated/{}.png".format(i)
        Image.fromarray(img).save(path)
        paths.append(path + "?{}".format(secrets.token_urlsafe(16)))

    return flask.jsonify(result=paths)


@app.route('/blend', methods=['POST'])
def blend():
    z_src = xp.array([request.json["z_src"]], dtype=xp.float32)
    z_ref = xp.array([request.json["z_ref"]], dtype=xp.float32)
    c = xp.array([request.json["c"]], dtype=xp.int32)
    mask_bin = base64.b64decode(request.json["mask"])

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    with open("tmp/mask.png", "wb") as f:
        f.write(bytearray(mask_bin))

    mask = np.array(Image.open("tmp/mask.png"))
    mask = np.array(mask[:,:,0] > 0, dtype=np.float32).reshape((1, mask.shape[0], mask.shape[1]))

    lmds = list(map(float, request.json["lambda"]))

    interpolation = 16
    generated_images = gen_blended_images(z_src, z_ref, c, mask, lmds, interpolation=interpolation)
    paths = []
    if not os.path.exists("static/demo_feature_blending/generated"):
        os.mkdir("static/demo_feature_blending/generated")
    for i in range(interpolation):
        path = "static/demo_feature_blending/generated/result{}.png".format(i)
        Image.fromarray(generated_images[i]).save(path)
        paths.append(path + "?{}".format(secrets.token_urlsafe(16)))

    return flask.jsonify(result=paths)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
