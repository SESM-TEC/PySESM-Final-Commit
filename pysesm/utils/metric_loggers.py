def dict_layer_hook(info):
    print(f"DictLayer - Época: {info['epoch']}, Pérdida: {info['loss']}")

def ista_layer_hook(info):
    print(f"ISTALayer - h: {info['h']}, Pérdida: {info['loss']}")

def sesm_hook(info):
    print(f"SESM - Época: {info['epoch']}, Pérdida ISTA: {info['loss_ista']}, Pérdida Dict: {info['loss_dict']}")