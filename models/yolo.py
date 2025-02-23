import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv, Bottleneck, SPP, SPPCSP, VoVCSP, DWConv, Focus, BottleneckCSP, BottleneckCSPLG, BottleneckCSPSE, BottleneckCSPSAM, BottleneckCSPSEA, BottleneckCSPSAMA, BottleneckCSPSAMB, BottleneckCSPGC, BottleneckCSPDNL, BottleneckCSP2, BottleneckCSP2SAM, Concat, DownC, DownD, DNL, GC, SAM, SAMA, NMS, autoShape, TR, BottleneckCSPTR, BottleneckCSP2TR, SPPCSPTR, ReOrg, BottleneckCSPF, ImplicitA, ImplicitM, DWT, RepVGGBlock, ConvRepVGG, BottleneckCSPRepVGG2, BottleneckCSPRepVGG3, BottleneckCSPFRepVGG, BottleneckCSPFRepVGG2, BottleneckRepVGG
from models.experimental import MixConv2d, CrossConv, C3
from utils.loss import SigmoidBin
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), rotated=False, ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.rotated = rotated
        if self.rotated:
            self.no += 1
        print(f"\n\n self.no = {self.no}, self.rotated = {self.rotated} \n\n")
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                if not hasattr(self, 'rotated'):
                    self.rotated = False
                if self.rotated:
                    y[..., 4:5] = y[..., 4:5] * 4. - 2. # angle = (-2.0 ; +2.0), use only [-1.0; +1.0] -> [-pi; +pi]

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), rotated=False, ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor

        #self.x_bin_sigmoid = SigmoidBin(bin_count=11, min=-0.5, max=1.5)
        #self.y_bin_sigmoid = SigmoidBin(bin_count=11, min=-0.5, max=1.5)
        self.w_bin_sigmoid = SigmoidBin(bin_count=21, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=21, min=0.0, max=4.0)
        # classes, x,y,obj
        self.no = nc + 3 + \
            self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()   # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()
            

        self.rotated = rotated
        if self.rotated:
            self.angle_bin_sigmoid = SigmoidBin(bin_count=31, min=-1.1, max=1.1)
            self.no += self.angle_bin_sigmoid.get_length() + 2      # im, re, angle-bce

        print(f"\n\n self.no = {self.no}, self.rotated = {self.rotated} \n\n")
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        #self.x_bin_sigmoid.use_fw_regression = True
        #self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True

        if not hasattr(self, 'rotated'):
            self.rotated = False

        if self.rotated:
            self.angle_bin_sigmoid.use_fw_regression = False

        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                #print(f" y[..., 0:12] = {y[..., 0:12].shape}, grid[i] = {self.grid[i].shape}, stride[i] = {self.stride[i].shape}, anchor_grid[i] = {self.anchor_grid[i].shape}")

                #print(f" self.stride[i] = {self.stride[i]}")

                #px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                #py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                #y[..., 0] = px
                #y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph

                if self.rotated:
                    angle_bias = self.angle_bin_sigmoid.forward(y[..., 48:80]) # 32 values = (1-reg + 31-bce)                                        
                    
                    im_bias = torch.sin( angle_bias * math.pi )
                    re_bias = torch.cos( angle_bias * math.pi )

                    y[..., 46:48] = y[..., 46:48] * 0.5 - 0.25
                    y[..., 4] = torch.atan2(y[..., 46] + im_bias, y[..., 47] + re_bias) / math.pi                    

                    y = torch.cat((y[..., 0:5], y[..., 80:]), dim=-1)
                else:
                    y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)

                #print(f" y = {y.shape}, y.view(bs, -1, y.shape[-1]) = {y.view(bs, -1, y.shape[-1]).shape}")
                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, nid=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nid:
            self.nid = nid
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        self.rotated = False
        if 'rotated' in self.yaml:
            self.rotated = self.yaml['rotated']
        print(f" cfg: rotated = {self.rotated}")

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IDetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect):
                    break

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0,1,2,24,46,47,48)].data
            obj_idx = 45 if m.rotated else 26
            b[:, :obj_idx].data += math.log(0.6 / (11 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0,1,2,24,46,47,48)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepVGGBlock):
                #print(f"fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    try:
        nid = d['nid']
    except:
        pass
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, BottleneckCSPLG, BottleneckCSPSE, BottleneckCSPSAM, BottleneckCSPSEA, BottleneckCSPSAMA, BottleneckCSPSAMB, BottleneckCSPGC, BottleneckCSPDNL, BottleneckCSP2, BottleneckCSP2SAM, SPPCSP, VoVCSP, C3, DownD, DownC, DNL, GC, SAM, SAMA, BottleneckCSPTR, BottleneckCSP2TR, TR, SPPCSPTR, BottleneckCSPF, RepVGGBlock, ConvRepVGG, BottleneckCSPRepVGG2, BottleneckCSPRepVGG3, BottleneckCSPFRepVGG, BottleneckCSPFRepVGG2, BottleneckRepVGG]:
            c1, c2 = ch[f if f <= -1 else f + 1], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSPLG, BottleneckCSPSE, BottleneckCSPSAM, BottleneckCSPSEA, BottleneckCSPSAMA, BottleneckCSPSAMB, BottleneckCSPGC, BottleneckCSPDNL, BottleneckCSP2, BottleneckCSP2SAM, SPPCSP, VoVCSP, C3, DownD, DownC, BottleneckCSPTR, BottleneckCSP2TR, TR, SPPCSPTR, BottleneckCSPF, BottleneckCSPRepVGG2, BottleneckCSPRepVGG3, BottleneckCSPFRepVGG, BottleneckCSPFRepVGG2]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f if f <= -1 else f + 1]]
        elif m is Concat:
            c2 = sum([ch[x if x <= -1 else x + 1] for x in f])
        elif m in [Detect, IDetect]:
            if len(args) == 2:
                args.append(False)  # rotated=False
            args.append([ch[x if x <= -1 else x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [ReOrg, DWT]:
            c2 = ch[f] * 4
        else:
            c2 = ch[f if f <= -1 else f + 1]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
