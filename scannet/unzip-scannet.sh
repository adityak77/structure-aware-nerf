
# get first argument corresponding to scannet scans folder
scans_folder=$1

# loop through all directories in scans folder and unzip all zip files in each directory and remove zip files
for dir in $scans_folder/*; do
    for file in $dir/*.zip; do
        unzip $file -d $dir
        rm $file
    done
done