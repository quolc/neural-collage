import os, sys, time
import shutil
import yaml

import argparse
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from evaluation import sample_reconstruction
import source.yaml_utils as yaml_utils


def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)
    copy_to_result_dir(
        config.models['generator']['fn'], result_dir)
    copy_to_result_dir(
        config.models['discriminator']['fn'], result_dir)
    copy_to_result_dir(
        config.models['encoder']['fn'], result_dir)
    copy_to_result_dir(
        config.dataset['dataset_fn'], result_dir)
    copy_to_result_dir(
        config.updater['fn'], result_dir)


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    enc_conf = config.models['encoder']
    enc = yaml_utils.load_model(enc_conf['fn'], enc_conf['name'], enc_conf['args'])
    return gen, dis, enc


def make_optimizer(model, alpha=0.0002, beta1=0., beta2=0.9):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--input_dir', type=str, default='./data/imagenet')
    parser.add_argument('--truth_dir', type=str, default='./data/imagenet')
    parser.add_argument('--results_dir', type=str, default='./results/gans',
                        help='directory to save the results to')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--gen_model', type=str, default='',
                        help='path to the generator .npz file')
    parser.add_argument('--dis_model', type=str, default='',
                        help='path to the discriminator .npz file')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data loading processes')

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    chainer.cuda.get_device_from_id(args.gpu).use()
    gen, dis, enc = load_models(config)

    chainer.serializers.load_npz(args.gen_model, gen)
    chainer.serializers.load_npz(args.dis_model, dis)

    gen.to_gpu(device=args.gpu)
    dis.to_gpu(device=args.gpu)
    enc.to_gpu(device=args.gpu)
    models = {"gen": gen, "dis": dis, "enc": enc}

    opt_enc = make_optimizer(
        enc, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opts = {"opt_enc": opt_enc}
    # Dataset
    config['dataset']['args']['root_input'] = args.input_dir
    config['dataset']['args']['root_truth'] = args.truth_dir
    dataset = yaml_utils.load_dataset(config)
    # Iterator
    iterator = chainer.iterators.MultiprocessIterator(
        dataset, config.batchsize, n_processes=args.loaderjob)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': iterator,
        'optimizer': opts,
    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ["loss", "min_slope", "max_slope", "min_z", "max_z"]
    # Set up logging
    trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))
    for m in models.values():
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))

    trainer.extend(sample_reconstruction(enc, gen, out, n_classes=gen.n_classes),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))
    ext_opt_enc = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_enc)
    trainer.extend(ext_opt_enc)
    if args.snapshot:
        print("Resume training with snapshot:{}".format(args.snapshot))
        chainer.serializers.load_npz(args.snapshot, trainer)

    # Run the training
    print("start training")
    trainer.run()


if __name__ == '__main__':
    main()
