//RUN: %cml -emit-ir -cpp %s -o -| FileCheck %s
#include<stdio.h>
#include<vector>

int main()
{
 std::vector<int> myVector;
 myVector.insert(myVector.end(),{1,2,3}); 
 myVector.insert(myVector.end(),{4,5,6}); 

 printf("%d %d %d\n", myVector[0],myVector[1],myVector[2]);
 
 printf("%d %d %d\n", myVector[3],myVector[4],myVector[5]);
  
 return 0;
}

//CHECK: cil.alloca !cil.class<std::__1::__wrap_iter>
//CHECK: _ZNSt3__16vectorIiNS_9allocatorIiEEE6insertENS_11__wrap_iterIPKiEESt16initializer_listIiE
