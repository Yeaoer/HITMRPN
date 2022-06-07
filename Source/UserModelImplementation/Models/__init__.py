# -*- coding: utf-8 -*
import JackFramework as jf

from .HITMRPN.inference import HITMRPNInterface


def model_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('HITMRPN'):
            jf.log.info("Enter the HITMRPN model")
            model = HITMRPNInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
