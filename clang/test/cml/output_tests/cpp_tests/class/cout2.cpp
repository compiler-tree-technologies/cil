// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
// XFAIL: *
#include <iostream>

int main() {
 std::cout << " Finally \n"; // CHECK: Finally
  return 0;
}
