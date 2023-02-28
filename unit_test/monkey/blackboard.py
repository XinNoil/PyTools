import mtools.monkey as mk

device = mk.get_free_gpu()
print(device)
mk.set_current_gpu(device)
print(mk.get_current_gpu())