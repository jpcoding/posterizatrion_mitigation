from netCDF4 import Dataset as ncdfDataset
import numpy as np
import sys,os

def bin2nc(filename, data,r1,r2,r3):
    dataset = ncdfDataset(filename, 'w', format='NETCDF4_CLASSIC')
    dataset.createDimension('x', r1)
    dataset.createDimension('y', r2)
    dataset.createDimension('z', r3)
    dataset.createVariable('velocity_x', np.float32, ('x', 'y', 'z'))
    velocity_x = data.reshape([r1, r2, r3])
    # print(dataset.variables['velocity_x'].shape)
    dataset.variables['velocity_x'][:] = velocity_x
    # print(dataset.variables['velocity_x'].shape)
    # print(dataset.variables['velocity_x'].dtype)
    # print(np.max(dataset.variables['velocity_x']))
    dataset.close()


# filename=sys.argv[1]
# basename=os.path.basename(filename)
data = np.fromfile(sys.argv[1],dtype=np.float32)
bin2nc(sys.argv[2], data, int(sys.argv[5]),int(sys.argv[4]),int(sys.argv[3]))