// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  unsigned short k;
  k = 10;
  // CHECK: 10
  printf("%u\n", k);
  return 0;
}
