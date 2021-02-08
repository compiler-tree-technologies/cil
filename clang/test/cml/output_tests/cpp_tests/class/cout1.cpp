// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <iostream>

int main() {
  std::cout << 10; // CHECK: 10
  return 0;
}
