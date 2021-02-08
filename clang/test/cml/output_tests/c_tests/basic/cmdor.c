// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 10;
  // CHECK: 10
  printf("%d\n", a);
  a |= -1;
  // CHECK: -1
  printf("%d\n", a);
  return 0;
}
