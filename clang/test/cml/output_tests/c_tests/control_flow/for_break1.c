// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3;
  for (int i = 0; i < 10; i++) {
    if (a >10) {
      break;
    }
    a  = a + 10;
  }
  printf("%d\n", a); // CHECK: 13
  return 0;
}
