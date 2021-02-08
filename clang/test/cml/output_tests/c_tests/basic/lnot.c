// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 1;
  if (!a)
    printf("FALSE");
  else
    // CHECK: TRUE
    printf("TRUE");
  return 0;
}
