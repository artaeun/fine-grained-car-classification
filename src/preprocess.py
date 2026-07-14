# pre-resize images to accelerate training
# run once before training to create image_256 folder
# supports gpu acceleration with USE_CUDA = True flag
# using PIL from cpu and torchvision for gpu downscaling

from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_jpeg

# THIS NEEDS TO BE DONE ONLY ONCE
# WITH FRESH DATASET FROM THE COMPCARS DOWNLOAD LINK

# Also, this processes both the wwhole sv dataset as well as the split dataset as per the original paper (and directory structure)


# Configuration
INPUT_DIR = 'dataset/data/image'
OUTPUT_DIR = 'dataset/data/image_256'
TARGET_SIZE = 256
QUALITY = 95 # in percentage
USE_CUDA = True # False for CPU (if you don't have CUDA or a nvdia GPU) 
BATCH_SIZE = 32 # ideal value for my 12GB GPU
VERIFY_ONLY = False # use this to skip processing and just verify that all images were done correctly


#Preprocessing functions

# basic idea: read image, resize to target size while making sure to handle different channel counts (grayscale, rgba), and then save to output directory with same relative path as input

def preprocess_dataset_cpu(input_dir, output_dir, target_size, quality):
    # pre-resize all images using cpu (pil)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    image_files = list(input_path.rglob("*.jpg"))# find all jpg images
    total_images = len(image_files)

    print(f"found {total_images} images to process")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print("using: cpu (pil)")

    processed_count = 0 # counters
    skipped_count = 0
    error_count = 0

    for img_file in tqdm(image_files, desc="resizing images"):
        # preserve directory structure when saving to output
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path

        # skip if already processed
        if output_file.exists():
            skipped_count = skipped_count + 1
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # resize and save
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((target_size, target_size), Image.BILINEAR) # biliniear is good for downscaling
        img.save(output_file, quality=quality)
        processed_count = processed_count + 1

    print(f"\nprocessing complete:")
    print(f"  processed: {processed_count}")
    print(f"  skipped (already exist): {skipped_count}")
    print(f"  errors: {error_count}")


def preprocess_dataset_gpu(input_dir, output_dir, target_size, quality, batch_size):
    # pre-resize all images using gpu (torchvision in this case instead of pil)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # check cuda availability
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("cuda not available, falling back to cpu")
        preprocess_dataset_cpu(input_dir, output_dir, target_size, quality)
        return

    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"using: gpu ({device_name})")

    # find all jpg images
    image_files = list(input_path.rglob("*.jpg"))
    total_images = len(image_files)

    print(f"found {total_images} images to process")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"batch size: {batch_size}")

    # filter to only unprocessed images
    work_items = []
    skipped_count = 0

    for img_file in image_files:
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path

        already_exists = output_file.exists()
        if already_exists:
            skipped_count = skipped_count + 1
            continue

        work_item = (img_file, output_file)
        work_items.append(work_item)

    print(f"skipped (already exist): {skipped_count}")
    print(f"to process: {len(work_items)}")

    num_work_items = len(work_items)
    if num_work_items == 0:
        print("nothing to process")
        return

    processed_count = 0
    error_count = 0

    # process in batches
    for batch_start in tqdm(range(0, num_work_items, batch_size), desc="processing batches"):
        batch_end = batch_start + batch_size # calculate batch end index
        if batch_end > num_work_items:
            batch_end = num_work_items

        batch_items = work_items[batch_start:batch_end]

        for img_file, output_file in batch_items:
            output_file.parent.mkdir(parents=True, exist_ok=True)

            img_tensor = read_image(str(img_file))  # read image as tensor [c, h, w]

            # handling of  different channel counts
            num_channels = img_tensor.shape[0]
            if num_channels == 1:
                # grayscale to rgb
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif num_channels == 4:
                # rgba to rgb (drop alpha, basically)
                img_tensor = img_tensor[:3, :, :]

            # move to gpu
            img_batch = img_tensor.unsqueeze(0)
            img_float = img_batch.float()
            img_gpu = img_float.to(device)

            # resize on gpu step
            resize_size = (target_size, target_size)
            resized = F.interpolate(
                img_gpu,
                size=resize_size,
                mode='bilinear',
                align_corners=False
            )

            # convert back to uint8 and move to cpu
            resized = resized.squeeze(0) # remove batch dimension
            resized = resized.clamp(0, 255) # ensure values are in valid range after interpolation
            resized = resized.byte() 
            resized = resized.cpu() # move back to cpu for saving with torchvision

            # save output
            output_path_str = str(output_file)
            write_jpeg(resized, output_path_str, quality=quality)
            processed_count = processed_count + 1

    print(f"\nprocessing complete:")
    print(f"  processed: {processed_count}")
    print(f"  skipped (already exist): {skipped_count}")
    print(f"  errors: {error_count}")


def verify_preprocessing(input_dir, output_dir):
    # verify that all images were processed correctly by comparing input and output directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # get input images
    input_images = set()
    for img_file in input_path.rglob("*.jpg"):
        relative_path = img_file.relative_to(input_path)
        input_images.add(relative_path)

    #get output images
    output_images = set()
    for img_file in output_path.rglob("*.jpg"):
        relative_path = img_file.relative_to(output_path)
        output_images.add(relative_path)

    # find anomaly in output vs input
    missing = input_images - output_images
    extra = output_images - input_images

    num_missing = len(missing)
    if num_missing > 0:
        print(f"missing {num_missing} images in output")
        missing_list = list(missing)
        for i in range(min(5, num_missing)):
            print(f"  {missing_list[i]}")
        is_valid = False
        return is_valid

    num_extra = len(extra)
    if num_extra > 0:
        print(f"found {num_extra} extra images in output")

    num_output = len(output_images)
    print(f"verification passed: {num_output} images")

    is_valid = True
    return is_valid


if __name__ == '__main__':
    # run based on configuration
    if VERIFY_ONLY:
        verify_preprocessing(INPUT_DIR, OUTPUT_DIR)
    elif USE_CUDA:
        preprocess_dataset_gpu(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            target_size=TARGET_SIZE,
            quality=QUALITY,
            batch_size=BATCH_SIZE
        )
        print("\nverifying preprocessing...")
        verify_preprocessing(INPUT_DIR, OUTPUT_DIR)
    else:
        preprocess_dataset_cpu(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            target_size=TARGET_SIZE,
            quality=QUALITY
        )
        print("\nverifying preprocessing...")
        verify_preprocessing(INPUT_DIR, OUTPUT_DIR)
