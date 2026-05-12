from adaptive_l1.data.utils import (
    create_data_split,
)

# set your path containing the fastMRI .h5 files here
files_path = "my/path/containing/fastMRI/files/"
split_path = "my/path/where/split/files/should/be/saved"
create_data_split(files_path, split_path)
