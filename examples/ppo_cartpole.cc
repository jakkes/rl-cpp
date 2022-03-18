#include <argparse/argparse.hpp>
#include <rl/rl.h>


using namespace rl;

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};
    parser
        .add_argument("duration")
        .help("Train duration, seconds.")
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    return parser;
}

int main(int argc, char **argv)
{

}
