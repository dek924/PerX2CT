def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value