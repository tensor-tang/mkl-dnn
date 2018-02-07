#!/bin/bash

export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export KMP_AFFINITY="granularity=fine,compact,0,0"

with_fuse=$1

if [ ! $with_fuse ]; then
	with_fuse=1
fi

iter=10

sum=0
for i in `seq 1 ${iter}`;
do
#	echo $i
#	./build/tests/gtests/test_convolution_forward_u8s8s32 | grep "conv relu"
	
	ms=`./build/examples/test-fuse $with_fuse 100 100 | grep ms | awk -F ':' '{print $2}' |  awk -F ' ' '{print $1}'`
	echo $ms
	sum=`awk 'BEGIN{printf "%f",('$sum' + '$ms')}'`
done

avg=`awk 'BEGIN{printf "%f",('$sum' / '$iter')}'`
echo "Avg of ${iter} : $avg" 
