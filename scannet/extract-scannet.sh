
# get first argument corresponding to scannet scans folder
scans_folder=$1

# loop through all directories in scans folder
for dir in $scans_folder/*; do
    echo "Processing $dir"

    # unzip all zip files in each directory and remove zip files
    for file in $dir/*.zip; do
        unzip -q $file -d $dir
    done

    # extract data from .sens files
    for file in $dir/*.sens; do
        python ScanNet/SensReader/python/reader.py --filename $file --output_path $dir --export_color_images --export_pose --export_intrinsics --frame_skip 10
    done
done