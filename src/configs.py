import hashlib
import os


class ConfigManager:
    def __init__(self, lab_path):
        self.configs = {}
        self.active = {}
        self.lab_path = lab_path

    def add(self, name, model, loss_fn, optim, batch_size, epochs):
        self.configs[name] = {
            "model": model,
            "loss_fn": loss_fn,
            "optim": optim,
            "batch_size": batch_size,
            "epochs": epochs,
        }
        return self

    def __repr__(self):
        return f"{self.configs}"

    def __getitem__(self, key):
        return self.configs[key]

    def __setitem__(self, key, value):
        self.configs[key] = value

    def activate(self, name):
        self.active = self.configs[name]

        active = self.active

        model = active["model"]
        optimizer = active["optim"]
        batch_size = active["batch_size"]
        num_epochs = active["epochs"]
        loss_fn = active["loss_fn"]

        active["checkpoint"] = {
            "info": {
                "model_name": model.__class__.__name__,
                "optimizer_name": optimizer.__class__.__name__,
                "loss_fn_name": loss_fn.__name__,
                "batch_size": batch_size,
                "epochs": num_epochs,
            },
            "optim_state": optimizer.state_dict(),
        }

        checkpoint = active["checkpoint"]

        checkpoint_str = str(checkpoint)
        hash_object = hashlib.md5(checkpoint_str.encode())
        hash = hash_object.hexdigest()[:10]

        model_path = os.path.join(self.lab_path, "model")

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        checkpoint_name = f"{checkpoint['info']['model_name']}_{hash}"
        checkpoint_file = f"{checkpoint_name}.pth"
        checkpoint_path = os.path.join(model_path, checkpoint_file)

        print(f"Checkpoint: {checkpoint_name}")

        active["checkpoint_path"] = checkpoint_path
