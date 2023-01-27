import abc
import torch


class CUDAGraphComputation(abc.ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.graph = None

    @abc.abstractmethod
    def forward(self, *x: torch.Tensor):
        pass

    def _init_graph(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.forward(*self.inputs)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.CUDAGraph(self.graph):
            self.outputs = self.forward(*self.inputs)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.inputs = inputs
            self._init_graph()
        
        for fwd_input, graph_input in zip(inputs, self.inputs):
            graph_input.copy_(fwd_input)
        self.graph.replay()
        return [graph_output.clone() for graph_output in self.outputs]


if __name__ == "__main__":
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        x = torch.randn(25, device="cuda")
        i = torch.arange(25, device="cuda")
        y = x.square()
        y.put_(i, x, accumulate=True)
    torch.cuda.current_stream().wait_stream(s)
    
    x.copy_(torch.randn(25, device="cuda"))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = x.square()
        y.put_(i, x, accumulate=True)

    for _ in range(3):
        x.copy_(torch.randn(25, device="cuda"))
        g.replay()
        print(y)
