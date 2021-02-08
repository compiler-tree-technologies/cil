//RUN: %cml -emit-ir -cpp %s -o -| FileCheck %s -check-prefix=OPTNONE
//RUN: %cml -optStl -emit-ir -cpp %s -o -| FileCheck %s -check-prefix=OPT
#include<stdio.h>
#include<vector>

int main()
{
 std::vector<int> myVector;
 myVector.push_back(1);
 myVector.push_back(2);
 myVector.push_back(3);
 
 myVector.insert(myVector.end(),{4,5,6}); 

 printf("%d %d %d\n", myVector[0],myVector[1],myVector[2]);
 
 printf("%d %d %d\n", myVector[3],myVector[4],myVector[5]);
  
 return 0;
}

//OPTNONE: @_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi::@vector
//OPTNONE: @_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi::@vector
//OPTNONE: @_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi::@vector


//OPT: cil.alloca !cil.class<std::__1::__wrap_iter> 
//OPT: @_ZNSt3__16vectorIiNS_9allocatorIiEEE6insertENS_11__wrap_iterIPKiEESt16initializer_listIiE::@vector
//OPT: @_ZNSt3__16vectorIiNS_9allocatorIiEEE6insertENS_11__wrap_iterIPKiEESt16initializer_listIiE::@vector
