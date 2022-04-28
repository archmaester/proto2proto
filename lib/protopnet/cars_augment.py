import Augmentor
import os
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = './datasets/cars/'
dir = datasets_root_dir + 'train/'
target_dir_name = 'train_augmented/'

# You will find the train_crop_augmented inside train_crop because of Augmentor. Move it up a directory.
# Source path
fd = dir
# Target directory name, Target path is set to os.path.join(fd, tfd) by Augmentor
tfd = target_dir_name

# Rotate
p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
p.flip_left_right(probability=0.5)
for i in range(10):
    p.process()
del p

# Skew
p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
p.skew(probability=1, magnitude=0.2)  # max 45 degrees
p.flip_left_right(probability=0.5)
for i in range(10):
    p.process()
del p

# Shear
p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
p.shear(probability=1, max_shear_left=10, max_shear_right=10)
p.flip_left_right(probability=0.5)
for i in range(10):
    p.process()
del p

# distortion
p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=5)
p.flip_left_right(probability=0.5)
for i in range(10):
    p.process()
del p
