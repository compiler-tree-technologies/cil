// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int i, k = 2;
  for (k++, i = 0; i < k; i++)
    // CHECK: 0
    // CHECK: 1
    // CHECK: 2
    printf("%d\n", i);
  return 0;
}
