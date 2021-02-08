// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdint.h>
#include <stdio.h>

int main() {
  int64_t l = 9223372036854775807;
  // CHECK: 9223372036854775807
  printf("%ld\n", l);
  return 0;
}
