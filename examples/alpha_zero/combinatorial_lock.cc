#include <argparse/argparse.hpp>


argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};

    parser
        .add_argument("--dim")
        .required()
        .scan<'i', int>();

    parser
        .add_argument("--length")
        .required()
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
        return parser;
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }
}


int main(int argc, char **argv)
{
    auto args = parse_args(argc, argv);
}
