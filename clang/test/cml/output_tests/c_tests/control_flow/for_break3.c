// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3, b = 1;
  for (int i = 0; i < 2; ++i) {
    while (a < 30) {
      a = a + 10;
      if (a > 20) {
        break;
      }
      b = b + 1;
    }
  }
  printf("%d %d\n", a, b); // CHECK: 33 2
  return 0;
}
