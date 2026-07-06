import timm
model = timm.create_model("mobilevitv2_100", pretrained=False)
for name, mod in model.named_modules():
    if isinstance(mod, timm.layers.ConvNormAct):
        print(f"ConvNormAct: {name} | conv={getattr(mod, 'conv', None)} | bn={getattr(mod, 'bn', None)}")
    elif 'conv' in name.lower() or 'bn' in name.lower():
        print(f"{name}: {type(mod)}")
