import torch
import torchlogix


def get_model(args):
    """
    Select model from the architecture.
    It can be a difflogic model or a baseline model.
    """
    llkw = {
        "connections": args.connections,
        "implementation": args.implementation,
        "device": args.device,
        "parametrization": args.parametrization,
        "forward_sampling": args.forward_sampling,
        "temperature": args.temperature,
        "weight_init": args.weight_init,
    }
    model_cls = torchlogix.models.__dict__[args.architecture]
    model = model_cls(**llkw)

    model = model.to(llkw["device"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Scale learning rate by output_gate_factor when set (paper: B/L models use factor 2)
    effective_lr = args.learning_rate * getattr(args, "output_gate_factor", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=getattr(args, "weight_decay", 0.002),
    )

    return model, loss_fn, optimizer
