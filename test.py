import torch


if __name__ == "__main__":
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x = torch.randn(25, 5, device="cuda")
        x[x < 0] = -x[x < 0]
        y = x.square()
    torch.cuda.current_stream().wait_stream(s)
    
    x.copy_(torch.randn(25, 5, device="cuda"))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x[x < 0] = -x[x < 0]
        y = x.square()

    for _ in range(3):
        x.copy_(torch.randn(25, 5, device="cuda"))
        g.replay()
        print(y)
