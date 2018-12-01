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


@app.route('/')
def index():
    return render_template('demo_spatial_translation/index.html',
                           dataset_name=config['dataset']['dataset_name'])


# "#010FFF" -> (1, 15, 255)
def hex2val(hex):
    if len(hex) != 7:
        raise Exception("invalid hex")
    val = int(hex[1:], 16)
    return np.array([val >> 16, (val >> 8) & 255, val & 255])


def gen_morphed_images(z, base_class, palette, masks, interpolation=8):
    z = xp.broadcast_to(z, (interpolation, 128))

    sizes = [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256]
    ws = []
    for i_size, size in enumerate(sizes):
        w = xp.zeros((interpolation, size, size, gen.n_classes), dtype=xp.float32)
        w[:, :, :, base_class] = 1.0  # default class
        for i_mask in range(len(palette)):
            resized_mask = xp.array(resize(masks[i_mask], (size, size)).reshape((size, size)), dtype=xp.float32)
            # resized_mask = xp.array(img_masks[i_mask].resize((size, size))).astype(xp.float32) / 255
            for i in range(interpolation):
                weight = i / (interpolation - 1.0)
                # if i_size <= 0:
                #     weight = 0
                w[i, :, :, base_class] -= resized_mask * weight
                w[i, :, :, palette[i_mask]] = resized_mask * weight
        ws.append(chainer.Variable(w))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        x = gen.spatial_interpolation(z, ws)

    x = x.data
    if args.gpu >= 0:
        x = x.get()

    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8).transpose((0,2,3,1))
    return x


@app.route('/post', methods=['POST'])
def post():
    if request.method == 'POST':
        z = xp.array(request.json["z"], dtype=xp.float32)
        base_class = int(request.json["c"])
        palette = [int(c) for c in request.json["palette"]]
        colors = [hex2val(hex) for hex in request.json["colors"]]
        class_map_bin = base64.b64decode(request.json["class_map"])

        # temporarily save class-map
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        with open("tmp/classmap.png", "wb") as f:
            f.write(bytearray(class_map_bin))

        # load and split class-map
        class_map = np.array(Image.open("tmp/classmap.png"))[:,:,:3]

        masks = []
        for i in range(len(colors)):
            # select region
            # mask = np.array((class_map == colors[i]).all(axis=2), dtype=np.uint8) * 255
            mask = np.array((np.isclose(class_map, colors[i], atol=2.0)).all(axis=2), dtype=np.uint8) * 255

            # debug
            Image.fromarray(mask, mode="L").save("tmp/mask{}.png".format(i))

            mask = np.asarray(gaussian_filter(mask, 8), dtype=np.float32).reshape((1, mask.shape[0], mask.shape[1])) / 255.0
            masks.append(mask)

        interpolation = 16
        generated_images = gen_morphed_images(z, base_class, palette, masks, interpolation=interpolation)
        paths = []
        if not os.path.exists("static/demo_spatial_translation/generated"):
            os.mkdir("static/demo_spatial_translation/generated")
        for i in range(interpolation):
            path = "static/demo_spatial_translation/generated/{}.png".format(i)
            Image.fromarray(generated_images[i]).save(path)
            paths.append(path + "?{}".format(secrets.token_urlsafe(16)))

        return flask.jsonify(result=paths)
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
