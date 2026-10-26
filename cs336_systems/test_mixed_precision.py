import torch

def main():
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("FP32 accumulate:", s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("FP16 accumulate:", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16).float()
    print("FP32 accum + FP16 addend:", s)

if __name__ == "__main__":
    main()