for file in ../../Dataset/custom-imageai/*; do
    # Check if the file is a regular file and not a directory
    if [[ -f "$file" ]]; then
        # Run main.py on the file
        python gui.py "$file"
    fi
done
