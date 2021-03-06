import numpy as np
import xarray as xr

my_arr = np.random.rand(4)
print('og shape', np.shape(my_arr))
new_arr = my_arr[np.newaxis, ...]
print("shape", np.shape(new_arr))

da = xr.DataArray(
    new_arr,
    coords={'time' : [15], 'lon': np.array([358, 359, 0, 1])},
    dims=["time", "lon"])

print("after", da)

# opt = {
#         'FFT_type': 'RFFT',
#         'save_all': 'Yes',
#         'th_ref': 300.0,
#         'dx': 20.0,
#         'dy': 20.0,
#           }
#
# dx_in = float(opt['dx'])
# print(dx_in)

# a=np.array([[0, 1, 4, 6, 2, 5, 3, 7], [4, 8, 2, 7, 9, 6, 4, 5]])
# b=np.array([[4, 9, 2, 7, 9, 1, 7, 5], [0, 1, 4, 8, 2, 7, 3, 4]])
#
# a_3 = (a >= 3)

# np.save('test', a_3)
# a_where_3 = np.where(a >= 3)
#
# b_5 = (b>=5)
#
# # print("a gt 3", a[~a_3])
#
# print(a_3)
# print(b_5)
#
# c = np.logical_and(a_3, b_5)
# d = np.logical_or(a_3, b_5)
#
# print('c=', c)
#
# print('d=', d)

# print(10**(-5))




