// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
// XFAIL: *
#include <iostream>

int main() {
  int *val = new int;
  *val = 11;
  std::cout << *val << "\n"; // CHECK: 11
  delete val;
  return 0;
}
