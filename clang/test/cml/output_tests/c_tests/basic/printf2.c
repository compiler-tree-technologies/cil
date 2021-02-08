// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a, b;
  a = 31;
  b = 20;
  printf("%d %d\n",a, b);
  return 0;
}

// CHECK: 31 20
