while read p; do
      echo $p
      python run.py --input_image $p --output_image_shape 50,50
  done <example_list.txt
