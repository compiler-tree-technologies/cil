// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3, b = 1;
  while (a < 30) {
    a  = a + 10;
    if (a<20) {
      continue;
    }
    b = b +  1;
  }
  printf("%d %d\n", a, b); // CHECK: 33 3
  return 0;
}
