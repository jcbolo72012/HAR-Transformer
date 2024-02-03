from src.library import *

f = load_from_tsfile('data/MotionSenseHAR/MotionSenseHAR_TRAIN.ts', return_data_type='numpy3d')

print(type(f))
print(type(f[0]), f[0].shape)
print(type(f[1]), f[1].shape)
      
