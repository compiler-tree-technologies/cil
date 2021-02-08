// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  a = 3;
  int b = 10;

  while (a < 10) {
    b = b + 10;
    a = a + 1;
  }

  printf("%d %d\n", a, b);
  return 0;
}
// CHECK: 10 80
