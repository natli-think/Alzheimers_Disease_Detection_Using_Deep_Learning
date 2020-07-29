import preprocess

input_image = Preprocess()

#Call preprocess functions
input_image.skull_strip(src_path,dst_path)
input_image.bias_correction(dst_path,dst_path_1)
input_image.segmentation(dst_path_1,dst_path_2)
input_image.final_two_dimensional_image(dst_path_2,final_dst_path)
