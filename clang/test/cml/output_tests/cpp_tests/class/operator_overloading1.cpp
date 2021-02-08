// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class cout {
public:
  int dummy;
};

void operator<<(cout &obj, int val) { printf("%d \n", val); }

int main() {
  cout obj;
  obj << 10; // CHECK: 10
  return 0;
}
