import torch
ckpt_path = "./checkpoints/CNN_backbone_best_4_14_val_4_23_train.pt"
print(f"Test de chargement : {ckpt_path}")
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint chargé")
    if isinstance(checkpoint, dict):
        print("Clés du checkpoint :", checkpoint.keys())
    else:
        print("Type du checkpoint :", type(checkpoint))
except Exception as e:
    print("Erreur lors du chargement :", e) 