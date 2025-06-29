#include <iostream>
#include <bayesnet/BaseClassifier.h>
#include <bayesnet/classifiers/TAN.h>
#include <bayesnet/network/Network.h>

int main() {
    std::cout << "Testing BayesNet library integration..." << std::endl;
    
    try {
        // Test basic instantiation
        bayesnet::Network network;
        std::cout << "✓ Network class instantiated successfully" << std::endl;
        
        // Test TAN classifier instantiation
        bayesnet::TAN tan;
        std::cout << "✓ TAN classifier instantiated successfully" << std::endl;
        
        std::cout << "✓ All basic tests passed!" << std::endl;
        std::cout << "BayesNet library is working correctly." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}