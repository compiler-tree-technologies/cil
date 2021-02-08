// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int b = 0;
  int c = !(b);
  // CHECK: 1
  printf("%d\n", c);
  return 0;
}
