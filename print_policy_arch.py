import torch

def print_mlp_architecture_from_state_dict(pt_path):
    state_dict = torch.load(pt_path, map_location="cpu")
    print(f"Loaded state_dict from {pt_path}\n")
    layers = []
    for k, v in state_dict.items():
        if ".weight" in k and "classifier" in k:
            # Only look at Linear layers in the classifier
            in_features = v.shape[1]
            out_features = v.shape[0]
            layers.append((in_features, out_features, k))
    print("=== MLP Architecture (in_features -> out_features per layer) ===")
    for i, (inp, out, k) in enumerate(layers):
        print(f"Layer {i}: {inp} -> {out} ({k})")
    print(f"\nTotal layers: {len(layers)}")

if __name__ == "__main__":
    print_mlp_architecture_from_state_dict("sampled_policy.pt")
