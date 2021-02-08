// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
// XFAIL: *
#include <iostream>

int main() {
  int a = 10;
  float b = 12.43;
  std::cout << " Finally " << a << " " << b; // CHECK: 10 12.43
  return 0;
}
