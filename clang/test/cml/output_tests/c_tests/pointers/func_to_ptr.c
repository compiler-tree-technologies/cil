// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int add(int a, int b) { return a + b; }

void foo(int a, int b, int (*ptr)(int, int)) {
  // CHECK: 35
  printf(" %d \n", ptr(a, b));
}

int main() {
  int a = 10;
  int b = 25;
  foo(a, b, (int (*)(int, int))add);
  return 0;
}
