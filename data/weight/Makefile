compression_object.so: object2py.o
	g++ object2py.o -g -o compression_object.so -shared -fPIC -I /usr/include/python2.7 -I /usr/lib64/python2.7/site-packages/numpy-1.11.2-py2.7-linux-x86_64.egg/numpy/core/include  -L /usr/lib64/python2.7 -l boost_python
object2py.o:
	g++ -c object2py.cpp -g -o object2py.o -fPIC -I /usr/include/python2.7 -I /usr/lib64/python2.7/site-packages/numpy-1.11.2-py2.7-linux-x86_64.egg/numpy/core/include

clean:
	rm -rf object2py.o
	rm -rf compression_object.so
	rm -rf weight.dat 
