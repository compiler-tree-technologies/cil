// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  int a[5];
  int *ptr1 = &a[1];
  int *ptr2 = &a[4];
  int offset = ptr2 - ptr1;
  // CHECK: 3
  printf("%d\n", offset);
  return 0;
}
