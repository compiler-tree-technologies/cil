// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int i = 5;
  while (--i > 2) {
    // CHECK: 4
    // CHECK: 3
    printf("%d\n", i);
  }

  i = 5;
  while (i-- > 2) {
    // CHECK: 4
    // CHECK: 3
    // CHECK: 2
    printf("%d\n", i);
  }
  return 0;
}
