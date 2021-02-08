// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

int main() {
  int *arr;
  arr = (int *)malloc(sizeof(int) * 3);
  arr[0] = 1;
  arr[1] = 2;
  arr[2] = 2;
  // CHECK: 1
  printf("%d\n", arr[0]);
  arr++;
  // CHECK: 2
  printf("%d\n", arr[0]);
  return 0;
}
