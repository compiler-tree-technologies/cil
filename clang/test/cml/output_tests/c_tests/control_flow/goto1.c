// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {

  goto exit;
  // CHECK-NOT: HELLO
  printf("HELLO\n");
exit:
  // CHECK: exit
  printf("exit\n");
  return 0;
}
