// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

int main() {
  float *arr;
  arr = (float *)malloc(sizeof(float) * 3);
  arr[0] = 1;
  arr[1] = 2;
  arr[2] = 2;
  float *ptr = &arr[1];
  // CHECK: 2.000000
  printf("%f\n", *ptr);
  ptr--;
  // CHECK: 1.000000
  printf("%f\n", *ptr);
  free(arr);
  return 0;
}
