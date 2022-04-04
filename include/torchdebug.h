#ifndef RL_TORCHDEBUG_H_
#define RL_TORCHDEBUG_H_


#include <iostream>

#include <torch/torch.h>


void tprint(torch::Tensor &x) {
    std::cout << x << "\n";
}
void tprint(torch::Tensor &&x) { print(x); }

void tprinti(torch::Tensor &x, int i) { tprint(x.index({i})); }
void tprinti(torch::Tensor &&x, int i) { tprinti(x, i); }

void tprinti(torch::Tensor &x, int i, int j) { tprint(x.index({i, j})); }
void tprinti(torch::Tensor &&x, int i, int j) { tprinti(x, i, j); }

void tprinti(torch::Tensor &x, int i, int j, int k) { tprint(x.index({i, j, k})); }
void tprinti(torch::Tensor &&x, int i, int j, int k) { tprinti(x, i, j, k); }

void tprinti(torch::Tensor &x, int i, int j, int k, int l) { tprint(x.index({i, j, k, l})); }
void tprinti(torch::Tensor &&x, int i, int j, int k, int l) { tprinti(x, i, j, k, l); }

void tprinti(torch::Tensor &x, int i, int j, int k, int l, int m) { tprint(x.index({i, j, k, l, m})); }
void tprinti(torch::Tensor &&x, int i, int j, int k, int l, int m) { tprinti(x, i, j, k, l, m); }

void tprintmean(torch::Tensor &x) { tprint(x.mean()); }
void tprintmean(torch::Tensor &&x) { tprintmean(x); }

void tprintmean(torch::Tensor &x, int i) { tprint(x.mean(i)); }
void tprintmean(torch::Tensor &&x, int i) { tprintmean(x, i); }

#endif /* RL_TORCHDEBUG_H_ */
