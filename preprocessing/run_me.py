import preprocess

'''
Initialize the below variable to your desired locations

initial_path -> path where the 3D brain image is present. For eg : '/content/drive/brain.nii'

path_to_store_stripped_skull -> mention the path where you would like to store the result of skull stripping process

path_to_store_bias_corrected -> mention the path where you would like to store the result of bias correction process

path_to_store_segmented_img-> mention the path where you would like to store the result of segmentation process

destination_path -> path where you would like to store the final result of preprocess phase.

'''
input_image = Preprocess()
input_image.strip_the_skull(initial_path,path_to_store_stripped_skull)
input_image.get_noiseless_image(path_to_store_stripped_skull,path_to_store_bias_corrected)
input_image.do_segmentation(path_to_store_bias_corrected,path_to_store_segmented_img)
input_image.return_2D_image(path_to_store_segmented_img,destination_path)
