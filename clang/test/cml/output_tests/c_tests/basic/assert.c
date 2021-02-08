// RUN: %cml %s -o %t && %t | FileCheck %s
#include <assert.h>
#include <stdio.h>

int main() {
  int k = 10;
  assert(k <= 10);
  // CHECK: 10
  printf("%d\n", k);
  return 0;
}
