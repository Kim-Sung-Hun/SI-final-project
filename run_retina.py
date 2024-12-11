from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0,1,2,3,4,5,6]
    conf = retina_autoenc()
    train(conf, gpus=gpus)

    # # infer the latents for training the latent DPM
    # # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [1, 2]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    #gpus = [0]
    #conf = retina_autoenc_latent()
    #train(conf, gpus=gpus)

    # # need to first train the diffae autoencoding model & infer the latents
    # # this requires only a single GPU.
    # gpus = [0]
    # conf = ffhq128_autoenc_cls()
    # train_cls(conf, gpus=gpus)

    # # after this you can do the manipulation!
