#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cutensor.h>

const int block_size = 16;

/**
Read the file corresponding the the filename provided and extract the text out of as an
string and return the string. Exit the program in case the file does not exist.
*/
std::string parseTextFromFile(const std::string& filename) {
    // Open the file in read mode 
    std::ifstream file(filename, std::ios::in);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(-1);
    }

    // Read the entire contents into a string
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();

    // Close the file
    file.close();

    // Calculate and print the length of the text
    std::cout << "Total length of the text is " << text.size() << std::endl;
    
    return text;
}

// Function to read words from the file
// File is expected to contain one word per line.
std::vector<std::string> readAndSplitLines(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> lines;

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return lines;
}



// Function to create the stoi and itos mappings
// stoi mapping would map each character to a integer (index)
// itos is the reverse mapping of that.
void createMappings(const std::vector<char>& chars, std::map<char, int>& stoi, std::map<int, char>& itos) {
    for (size_t i = 0; i < chars.size(); ++i) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }
}

// Function to encode a string
std::vector<int> encode(const std::string& s, const std::map<char, int>& stoi) {
    std::vector<int> encoded;
    for (char c : s) {
        encoded.push_back(stoi.at(c));
    }

    // Print the encoded text
    //std::cout << "Encoded text: ";
    //for (int val : encoded) {
      //  std::cout << val << ' ';
   // }
    //std::cout << std::endl;
    return encoded;
}

// Function to decode a vector of integers
std::string decode(const std::vector<int>& l, const std::map<int, char>& itos) {
    std::string decoded;
    for (int i : l) {
        decoded.push_back(itos.at(i));
    }
    return decoded;
}

// Function to split data into training and validation sets
void splitData(const std::vector<int>& data, double train_ratio, std::vector<int>& train_data, std::vector<int>& val_data) {
    size_t n = static_cast<size_t>(train_ratio * data.size());
    train_data.assign(data.begin(), data.begin() + n);
    val_data.assign(data.begin() + n, data.end());
}


// Function to build dataset for training
std::pair<std::vector<std::vector<int>>, std::vector<int>> buildDataset(const std::vector<std::string>& words, const std::map<char, int>& stoi) {
    std::vector<std::vector<int>> X;
    std::vector<int> Y;
    for (const std::string& w : words) {
        std::vector<int> context(block_size, 0); // initialize context with zeros
        for (char ch : (w + '.')) {
            int ix = stoi.at(ch);
            X.push_back(context);
            Y.push_back(ix);
            context.erase(context.begin());
            context.push_back(ix);
        }
    }
    return {X, Y};
}

// Utility function to shuffle data
template <typename T>
void shuffleData(std::vector<T>& data) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}



int main() {
    printf("Hello World!!!");

    /**
    * Part 1: Prepare the training data. Extract the text from input file,
    * creat vocabulary (which is set of unique characters, also known as tokens) 
    * out of the text, 
    * Assign indexes to the vocabulary (tokens). 
    */
    std::string filename = "names.txt";

    // Read and split lines from the file
    std::vector<std::string> words = readAndSplitLines(filename);

    // Unique sorted characters and mappings
    std::set<char> charSet;
    for (const auto& w : words) {
        charSet.insert(w.begin(), w.end());
    }
    charSet.insert('.'); // Add start and end marker

    std::vector<char> chars(charSet.begin(), charSet.end());
    std::map<char, int> stoi;
    std::map<int, char> itos;
    createMappings(chars, stoi, itos);


    // Vectors to hold the training set
    std::vector<int> xs, ys;

    for (const std::string& w : words) {
        std::string chs = "." + w + "."; // Add start and end markers
        for (size_t i = 0; i < chs.size() - 1; ++i) {
            xs.push_back(stoi[chs[i]]);
            ys.push_back(stoi[chs[i + 1]]);
        }
    }

    // Number of elements
    size_t num_elements = xs.size();

    // Output for verification
    std::cout << "Number of elements: " << num_elements << std::endl;

    // Shuffle words
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(words.begin(), words.end(), g);

    // Split the data
    size_t n1 = static_cast<size_t>(0.8 * words.size());
    size_t n2 = static_cast<size_t>(0.9 * words.size());

    auto [Xtr, Ytr] = buildDataset({words.begin(), words.begin() + n1}, stoi);
    auto [Xdev, Ydev] = buildDataset({words.begin() + n1, words.begin() + n2}, stoi);
    auto [Xte, Yte] = buildDataset({words.begin() + n2, words.end()}, stoi);

    // Output for verification
    std::cout << "Training set size: " << Xtr.size() << ", " << Ytr.size() << std::endl;
    std::cout << "Validation set size: " << Xdev.size() << ", " << Ydev.size() << std::endl;
    std::cout << "Test set size: " << Xte.size() << ", " << Yte.size() << std::endl;

}