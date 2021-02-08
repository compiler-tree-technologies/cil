// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 10;
  int i = 0;
  int *ptr = &a;
  while (ptr) {
    // CHECK: 0
    // CHECK: 1
    // CHECK: 2
    printf("%d\n", i);
    i++;
    if (i == 3)
      ptr = NULL;
  }
  return 0;
}
