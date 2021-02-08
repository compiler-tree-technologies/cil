// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  const char action[3][32] = {"start", "stop", "resume"};
  // CHECK: start
  printf("%s\n", action[0]);
  // CHECK: stop
  printf("%s\n", action[1]);
  // CHECK: resume
  printf("%s\n", action[2]);
  return 0;
}
