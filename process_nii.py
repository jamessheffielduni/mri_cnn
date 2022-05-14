import nibabel
import nilearn
from nilearn import image
from nilearn.image import resample_img
import numpy as np

##Returns a np array containing all components
def resize(concatenated_file, afine_value):
    scan = nibabel.load(concatenated_file)
    volume = scan.get_fdata()
    lst=[]

    np.round(scan.affine)
    downscaledimage = resample_img(scan, target_affine=np.eye(3)*afine_value, interpolation='nearest')

    for slice_number in range(0, downscaledimage.shape[2]):
      lst.append(downscaledimage.dataobj[:,:,slice_number])
    slice_array = np.stack(lst, axis=2)

    return slice_array 

def resize_5D_to_4D(concatenated_file, afine_value):
    scan = nibabel.load(concatenated_file)
    volume = scan.get_fdata()
    lst=[]
    temp=[]

    np.round(scan.affine)
    downscaledimage = resample_img(scan, target_affine=np.eye(3)*afine_value, interpolation='nearest')

    for slice_number in range(0, downscaledimage.shape[2]):
      lst.append(downscaledimage.dataobj[:,:,slice_number])
    slice_array = np.stack(lst, axis=2)

    final_array = slice_array[:,:,30,:]
    
    return final_array 

def create_test_set(test_painful_scans, test_painless_scans):
  painful_labels_test = np.array([1 for _ in range(len(test_painful_scans))])
  painless_labels_test = np.array([0 for _ in range(len(test_painless_scans))])

  x_test = np.concatenate((test_painful_scans, test_painless_scans), axis=0)
  y_test = np.concatenate((painful_labels_test, painless_labels_test), axis=0)

  return x_test, y_test

def create_training_set(train_painful_scans, train_painless_scans):
  painful_labels_train = np.array([1 for _ in range(len(train_painful_scans))])
  painless_labels_train = np.array([0 for _ in range(len(train_painless_scans))])

  x_train = np.concatenate((train_painful_scans, train_painless_scans), axis=0)
  y_train = np.concatenate((painful_labels_train, painless_labels_train), axis=0)

  return x_train, y_train

def four_slice_average(concatenated_file, mri_view, excluded_slices_start, excluded_slices_end):
    scan = nibabel.load(concatenated_file)
    volume = scan.get_fdata()
    original_num_slices = 91
    num_slices = original_num_slices - excluded_slices_start - excluded_slices_end
    lst=[]
    lst_average=[]
    number_of_slices_to_average = 4

    #for slice_number in range(excluded_slices_start,original_num_slices-excluded_slices_end):
    #  lst.append(volume[:,:,slice_number])

    #########################################
    for slice_ in range(0, len(lst)-number_of_slices_to_average, number_of_slices_to_average):
      temp1 = lst[slice_]
      temp2 = lst[slice_ + 1]
      temp3 = lst[slice_ + 2]
      temp4 = lst[slice_ + 3]
      average = (temp1 + temp2 + temp3 + temp4) / number_of_slices_to_average
      lst_average.append(average)
    slice_array = np.stack(lst_average, axis=2)
    ##########################################

    #slice_array = np.stack(lst, axis=2)

    ##########################################

    return slice_array