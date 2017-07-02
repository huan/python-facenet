import numpy as np
a = np.arange(0, 60, 5)
a = a.reshape(2, 2, 3)

print('Original array is:')
print(a)
print('\n')

it = np.nditer(
    a,
    flags=['external_loop'],
    order='F',
)
print('Modified array is:')
for x in it:
    print('>', x),
