# -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import JackFramework as jf
import torch
from .model import HITMRPN
import torch.optim as optim
import torch.nn.functional as F
# import UserModelImplementation.user_define as user_def


class HITMRPNInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        args = self.__args
        # return model
        model = HITMRPN(args.templatesize, args.mode)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        # return opt and sch
        opt = optim.Adam(model[0].parameters(), lr=lr, betas=(0.5, 0.999),weight_decay=1e-4)
        def lambda_rule(epoch):
            return (1 - epoch / args.maxEpochs) ** 4
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule, last_epoch=-1)
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        sch.step()
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        # args = self.__args
        output = model(input_data[0], input_data[1])
        # return output
        return [output]

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        return []

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        label_gt = torch.ones([output_data[0].shape]).to(output_data[0].device)
        loss2 = F.cross_entropy(output_data[1], label_gt)
        loss1 = F.smooth_l1_loss(output_data[0], label_data[0])
        loss = 0.1 * loss1 + loss2
        return [loss]

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
