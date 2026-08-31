#include <inference.hpp>
#include <iostream>

int main() {
    inference::Model model = inference::Model("resources/model.gguf");
    inference::Tokenizer tokenizer = inference::Tokenizer();
    std::vector<int> tokens = tokenizer.encode("fuck you");
    for (int token : tokens) {
        std::cout << token << ' ';
    }
    std::cout << '\n';
    std::cout << tokenizer.decode(tokens) << '\n';
    return 0;
}
