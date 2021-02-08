// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 3;
  for (int i = 0; i < 10; i++) {
    if (a<10) {
      continue;
    }
    a  = a + 10;
  }
  printf("%d\n", a); // CHECK: 3
  return 0;
}
