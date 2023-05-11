path="../../Dataset/custom-videos"

for file in ${path}/*; do
    # Check if the file is a regular file and not a directory
    if [[ -f "$file" ]]; then
        # Run main.py on the file
        python FirstVideoObjectDetection.py "$file"
    fi
done
