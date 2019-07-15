import torch


'''
	# torch.Tensor.data_ptr()  :: return the address of the first element
	# x is y
	# 'is' checks identity. The same object / both stored in  the same memory address
	# '==' checks equality
'''
x = torch.randn(4, 4)
y = x.view(2,-1) # share memory
print(x.data_ptr() == y.data_ptr()) # prints True   x is y == False
y = x.clone().view(2,-1)
print(x.data_ptr() == y.data_ptr()) # prints False


x = torch.arange(10)
y = x[1::2]
print(x.data_ptr() == y.data_ptr()) # prints False for different data pointer



def same_storage(x, y):
	x_ptrs = set(e.data_ptr() for e in x.view(-1))
	y_ptrs = set(e.data_ptr() for e in y.view(-1))
	return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)

x = torch.arange(10)
y = x[1::2]
print(same_storage(x, y)) # prints True
z = y.clone()
print(same_storage(x, z)) # prints False
print(same_storage(y, z)) # prints False