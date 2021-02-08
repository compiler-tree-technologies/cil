// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int i, k = 2;
  int j = 0;
  for (k++, i = 0; i < k; i++, j += 1)
    // CHECK: 0 0
    // CHECK: 1 1
    // CHECK: 2 2
    printf("%d %d\n", i, j);

  i = k += j;
  // CHECK: 6
  printf("%d\n", i);
  return 0;
}
