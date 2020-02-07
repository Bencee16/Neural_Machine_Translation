import torch


def save_model(model, model_name, embedding_space, learning_rate, iterations):
    filename = f"{model_name}_{embedding_space}_{learning_rate}_{iterations}"
    torch.save(model.state_dict(), f"models/{filename}.pt")


def load_model_dict(model_name, embedding_space, learning_rate, iterations):
    filename = f"{model_name}_{embedding_space}_{learning_rate}_{iterations}"
    return torch.load(f"models/{filename}.pt")


def save_losses(filename, losses):
    with open(filename, "w") as file:
        for item in losses:
            file.write(f"{item}\n")