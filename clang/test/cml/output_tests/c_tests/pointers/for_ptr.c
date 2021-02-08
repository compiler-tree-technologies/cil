// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 5;
  int *p = &a;
  for (int i = 0; p; i++) {
    // CHECK: 1
    // CHECK: 2
    // CHECK: 3
    // CHECK: 4
    // CHECK: 5
    printf("%d\n", i);
    if (i == *p)
      p = NULL;
  }
  return 0;
}
