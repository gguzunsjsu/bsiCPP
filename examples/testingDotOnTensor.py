import torch

def dot_tensor():
    vec1 = torch.tensor([0.02, -0.04, -0.08, 0.16], dtype=torch.float32)
    vec2 = torch.tensor([-0.505343345, -0.12364533, 0.2162351, -0.512], dtype=torch.float32)
    expected = torch.dot(vec1, vec2).item()
    print(f"Vec1:  {vec1.tolist()}")
    print(f"Vec2:  {vec2.tolist()}")
    print(f"Expected dot product: {expected}")

dot_tensor()