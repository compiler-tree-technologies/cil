// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  a = 3;
  int c = 10;
  int b = 10;

  while (a < 10) {
    b = b + 10;
    a = a + 1;
  }
  while (c < 100) {
    c = c + 10;
  }

  printf("%d %d %d\n", a, b, c);
  return 0;
}
// CHECK: 10 80 100
