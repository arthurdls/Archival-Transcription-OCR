#!/bin/bash

#generate 90k "handwritten" lines
trdg -c 10000 -w 5 -f 96 -b 1 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_line &
trdg -c 10000 -w 5 -f 96 -b 1 -bl 2 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_blur_line &
trdg -c 10000 -w 5 -f 96 -b 1 -d 3 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_distort_line &
trdg -c 10000 -w 5 -f 96 -b 0 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_line &
trdg -c 10000 -w 5 -f 96 -b 0 -bl 2 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_blur_line &
trdg -c 10000 -w 5 -f 96 -b 0 -d 3 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_distort_line &
trdg -c 10000 -w 5 -f 96 -b 3 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_line &
trdg -c 10000 -w 5 -f 96 -b 3 -bl 2 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_blur_line &
trdg -c 10000 -w 5 -f 96 -b 3 -d 3 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_distort_line 


#generate 90k "handwritten" words
trdg -c 10000 -w 1 -f 96 -b 1 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_line &
trdg -c 10000 -w 1 -f 96 -b 1 -bl 2 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_blur_line &
trdg -c 10000 -w 1 -f 96 -b 1 -d 3 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/white_distort_line &
trdg -c 10000 -w 1 -f 96 -b 0 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_line &
trdg -c 10000 -w 1 -f 96 -b 0 -bl 2 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_blur_line &
trdg -c 10000 -w 1 -f 96 -b 0 -d 3 -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/noise_distort_line &
trdg -c 10000 -w 1 -f 96 -b 3 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_line &
trdg -c 10000 -w 1 -f 96 -b 3 -bl 2 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_blur_line &
trdg -c 10000 -w 1 -f 96 -b 3 -d 3 -id sepia -fd "handwritten_fonts" -na 2 --output_dir handwritten_lines/sepia_distort_line 

#generate 90k print lines
trdg -c 10000 -w 5 -f 96 -b 1 -na 2 --output_dir print_lines/white_line &
trdg -c 10000 -w 5 -f 96 -b 1 -bl 2 -na 2 --output_dir print_lines/white_blur_line &
trdg -c 10000 -w 5 -f 96 -b 1 -d 3 -na 2 --output_dir print_lines/white_distort_line &
trdg -c 10000 -w 5 -f 96 -b 0 -na 2 --output_dir print_lines/noise_line &
trdg -c 10000 -w 5 -f 96 -b 0 -bl 2 -na 2 --output_dir print_lines/noise_blur_line &
trdg -c 10000 -w 5 -f 96 -b 0 -d 3 -na 2 --output_dir print_lines/noise_distort_line &
trdg -c 10000 -w 5 -f 96 -b 3 -id sepia -na 2 --output_dir print_lines/sepia_line &
trdg -c 10000 -w 5 -f 96 -b 3 -bl 2 -id sepia -na 2 --output_dir print_lines/sepia_blur_line &
trdg -c 10000 -w 5 -f 96 -b 3 -d 3 -id sepia -na 2 --output_dir print_lines/sepia_distort_line 

#generate 90k print words
trdg -c 10000 -w 1 -f 96 -b 1 -na 2 --output_dir print_words/white_word &
trdg -c 10000 -w 1 -f 96 -b 1 -bl 2 -na 2 --output_dir print_words/white_blur_word &
trdg -c 10000 -w 1 -f 96 -b 1 -d 3 -na 2 --output_dir print_words/white_distort_word &
trdg -c 10000 -w 1 -f 96 -b 0 -na 2 --output_dir print_words/noise_word &
trdg -c 10000 -w 1 -f 96 -b 0 -bl 2 -na 2 --output_dir print_words/noise_blur_word &
trdg -c 10000 -w 1 -f 96 -b 0 -d 3 -na 2 --output_dir print_words/noise_distort_word &
trdg -c 10000 -w 1 -f 96 -b 3 -id sepia -na 2 --output_dir print_words/sepia_word &
trdg -c 10000 -w 1 -f 96 -b 3 -bl 2 -id sepia -na 2 --output_dir print_words/sepia_blur_word &
trdg -c 10000 -w 1 -f 96 -b 3 -d 3 -id sepia -na 2 --output_dir print_words/sepia_distort_word 