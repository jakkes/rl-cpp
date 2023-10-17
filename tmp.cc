#include <torch/torch.h>


class AbstractNet : public virtual torch::nn::Module
{
    public:
        virtual ~AbstractNet() = default;

        virtual torch::Tensor forward(const torch::Tensor &x) = 0;
};


class Net : public AbstractNet, public torch::nn::Cloneable<Net>
{
    public:
        Net() {
            reset();
        }

        void reset() override {
            fc1 = register_module("fc1", torch::nn::Linear(10, 10));
        }

        torch::Tensor forward(const torch::Tensor &x) override {
            return fc1->forward(x);
        }

    private:
        std::shared_ptr<torch::nn::LinearImpl> fc1;
};


int main()
{
    auto net = std::make_shared<Net>();
    auto net2 = std::dynamic_pointer_cast<AbstractNet>(net->clone());
    auto net3 = std::dynamic_pointer_cast<Net>(net2->clone());

    auto x = torch::randn({1, 10});
    auto y = net->forward(x);
    auto y2 = net2->forward(x);
    auto y3 = net3->forward(x);

    std::cout << y << std::endl;
    std::cout << y2 << std::endl;
    std::cout << y3 << std::endl;
}
