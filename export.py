import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from mit_semseg.config import cfg
from mit_semseg.models import ModelBuilder, SegmenetationModule

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SegWrapper(nn.Module):
    def __init__(self, seg, cfg):
        self.net = seg
        self.cfg = cfg

    def forward(self, X):
        ## Helper Functions
        def get_image_data(img, cfg):
            imgSizes = self.cfg.DATASET.imgSizes
            imgMaxSize = self.cfg.DATASET.imgMaxSize
            padding_constant = self.cfg.DATASET.padding_constant
            ori_height, ori_width, _ = img.size()

            def round2nearest_multiple(x, p):
                return ((x - 1) // p + 1) * p

            def img_transform(img):
                # 0-255 to 0-1
                img = img / 255.
                img = img.permute(2, 0, 1)
                trans = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                img = trans(img)
                return img

            img_resized_list = []
            for this_short_size in imgSizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            imgMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)

                # to avoid rounding in network
                target_width = round2nearest_multiple(target_width, padding_constant)
                target_height = round2nearest_multiple(target_height, padding_constant)

                # resize images
                # kind of hacky but prevents cast to np array
                img_cast = img.type(torch.uint8).float()
                img_resized = F.interpolate(img_cast.unsqueeze(0).permute(0, 3, 1, 2),
                                            size=(target_height, target_width), mode='bilinear')
                img_resized = img_resized[0].permute(1,2,0)
                img_resized = img_resized.type(torch.uint8)

                # image transform, to torch float tensor 3xHxW
                img_resized = img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)

            output = dict()
            output['img_ori'] = img
            output['img_data'] = [x.contiguous() for x in img_resized_list]
            output['info'] = '--img goes here'
            return output

        ## Forward Pass
        img_data = get_image_data(X, self.cfg)

        segSize = (img_data['img_ori'].shape[0],
                   img_data['img_ori'].shape[1])

        img_resized_list = img_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).cuda()

            for img in img_resized_list:
                feed_dict = img_data.copy()
                feed_dict['img_data'] = img.to(device)
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict['img_data'].to(device)
                # feed_dict = async_copy_to(feed_dict, device)

                # forward pass
                pred_tmp = self.net(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(self.cfg.DATASET.imgSizes)

        pred_scores, labels = torch.max(scores, dim=1)
        pred_scores = pred_scores.squeeze(0).cpu().detach()
        labels = labels.squeeze(0).cpu()

        return pred_scores, labels



def trace_input(cfg, im_path, device, savepath):
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)

    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_encoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.to(device)
    segmentation_module.eval()

    # The "black box" input X goes in, output Y comes out
    exnet = SegWrapper(segmentation_module, cfg)

    # image you will trace through the network
    img = Image.open(im_path).copnvert('RGB')
    img = torch.Tensor(np.array(img))

    traced_net = torch.jit.trace(exnet, img)
    traced_net.save(savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="trace trained segmentation module"
    )
    parser.add_argument(
        '--img',
        required=True,
        type='str',
        help='image used to trace segmentation (required)'
    )
    parser.add_argument(
        '--cfg',
        required=True,
        metavar='FILE',
        help='path to cfg file of model'
    )
    parser.add_argument(
        '--saveas',
        default='./exported_model.pth',
        type='str',
        help='target path for exported torchscript model'
    )

    chkpt_path = './' + args.cfg.DIR + '/'
    encoder_path = chkpt_path + 'encoder_' + args.cfg.TEST.checkpoint
    decoder_path = chkpt_path + 'decoder_' + args.cfg.TEST.checkpoint

    # modify these directly if you have issues
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.weights_encoder = encoder_path
    cfg.MODEL.weights_decoder = decoder_path 

    img_path = args.img
    trace_input(cfg, img_path, device, savepath=args.saveas)
    
    
