// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3;
  int b = 5;
  int c = 4;
  if (a >= 3)
    a++;

  for (int i = 0; i < 10; i++) {
    a = a + 3;
    if (b < 300) {
      for (int j = 0; j < 10; j++) {
        b = b + 5;
        while (b < 100) {
          c = c + 5;
          b++;
        }
      }
    }
  }
  printf("%d %d %d\n", a, b, c); // CHECK: 34 345 454
  return 0;
}
