// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  unsigned int a = 10;
  unsigned int b = 4;
  unsigned int c = a % b;
  // CHECK: 2
  printf("%u\n", c);
  return 0;
}
