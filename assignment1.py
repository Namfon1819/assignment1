import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_histogram(block):
    histogram = cv2.calcHist([block], [0], None, [256], [0, 256])
    return histogram

def reconstruct_image(input_image, block_size, overlap, num_blocks=4, output_directory="output"):
    height, width, channels = input_image.shape
    stride = block_size - overlap
    
    num_blocks_h = (height - overlap) // stride
    num_blocks_w = (width - overlap) // stride
    
    block_histograms = []  
    blocks = []  
    num_blocks_to_display = min(num_blocks, num_blocks_h * num_blocks_w)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    for i in range(num_blocks_h):
        if len(blocks) >= num_blocks_to_display:
            break
        for j in range(num_blocks_w):
            if len(blocks) >= num_blocks_to_display:
                break
            start_h = i * stride
            end_h = start_h + block_size
            
            start_w = j * stride
            end_w = start_w + block_size
            
            block = input_image[start_h:end_h, start_w:end_w]
            
            
            if channels == 1:
                histogram = compute_histogram(block)
            else:
                histogram = []
                for channel in range(channels):
                    channel_histogram = compute_histogram(block[:, :, channel])
                    histogram.extend(channel_histogram)
                histogram = np.array(histogram)
          
            block_histograms.append(histogram)
            blocks.append(block)
            
            # Save the block as an image
            output_path = os.path.join(output_directory, f'Block_{len(blocks)}.png')
            cv2.imwrite(output_path, block)
    
    return blocks, block_histograms

input_image = cv2.imread(r'C:\BLOCK\image.png')

block_size = 70
overlap = 25
num_blocks_to_display = 4 
output_directory = "output"  # Directory to save the blocks

blocks, block_histograms = reconstruct_image(input_image, block_size, overlap, num_blocks_to_display, output_directory)

for i, (block, histogram) in enumerate(zip(blocks, block_histograms), 1):
    plt.subplot(num_blocks_to_display, 2, i * 2 - 1)
    plt.imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
    plt.title(f'Block {i}')
    plt.axis('off')
    
    plt.subplot(num_blocks_to_display, 2, i * 2)
    plt.plot(histogram)
    plt.title(f'Histogram {i}')
    
plt.tight_layout()
plt.show()
