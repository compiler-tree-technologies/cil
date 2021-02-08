// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {

  int i = 0;
loop:
  i = i + 1;
  // CHECK: 1
  // CHECK: 2
  // CHECK: 3
  printf("%d\n", i);
  if (i == 3)
    goto exit;
  else
    goto loop;
exit:
  return 0;
}
