import sys
try:
    import torch
    print('torch.__version__:', torch.__version__)
    print('torch imported from:', torch.__file__)
except Exception as e:
    print('Erreur import torch:', e)
    sys.exit(1)

ckpt_path = './checkpoints/CNN_backbone_best_4_14_val_4_23_train.pt'
print(f'Test de chargement : {ckpt_path}')
try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print('Checkpoint chargé')
    if isinstance(checkpoint, dict):
        print('Clés du checkpoint :', checkpoint.keys())
    else:
        print('Type du checkpoint :', type(checkpoint))
except Exception as e:
    print('Erreur lors du chargement du checkpoint :', e) 