// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int b = 122;
  if (b)
    // CHECK: 122
    printf("%d\n", b);
  return 0;
}
