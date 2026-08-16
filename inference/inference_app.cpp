#include <inference.hpp>
#include <iostream>

int main() {
    inference::Model model = inference::Model(8, 8);
    inference::Tokenizer tokenizer = inference::Tokenizer();
    tokenizer.load("resources/tokenizer.txt");
    std::vector<int> tokens = tokenizer.encode("fuck you");
    for (int token : tokens) {
        std::cout << token << ' ';
    }
    std::cout << '\n';
    std::cout << tokenizer.decode(tokens) << '\n';
    return 0;
}
