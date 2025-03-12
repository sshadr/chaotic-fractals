import os
import numpy as np
import cv2
import imageio

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ldr_extensions = [".jpg", ".png"]
hdr_extensions = [".exr", ".hdr"]
#==============================================

# find all images in a directory and return list of full paths
def find_images(path, ext = ldr_extensions + hdr_extensions):
    if not isinstance(ext, list):
        ext = [ext]
    img_files = []
    for file in sorted(os.listdir(path)):
        if any([file.endswith(e) for e in ext]):
            img_files.append(os.path.join(path, file))   
    if not img_files:
        print("No images found in", path)
    return img_files

#==============================================

# load an image
def load_image(path, normalize=True, append_alpha=False):
    
    assert os.path.isfile(path), "Image file does not exist"
    is_hdr = is_hdr_from_file_extension(path)
    flags = (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) if is_hdr else cv2.IMREAD_UNCHANGED

    img = cv2.imread(path, flags)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize and not is_hdr:
        img = img.astype(np.float32) / 255.
    if append_alpha and img.shape[2] == 3:
        alpha = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, alpha], axis=-1)
    return img

#==============================================

# save an image
def save_image(img, path, channels=3, jpeg_quality=95):
    is_hdr = is_hdr_from_file_extension(path)

    if img.ndim == 2:
        out_img = img[..., None]
    if img.ndim == 3 and img.shape[2] >= 2:
        if channels == 2:
            out_img = np.zeros((*img.shape[0:2], 3))
            out_img[..., 1:3] = img[..., 2::-1]
        if channels == 3:
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if channels == 4:
            out_img = cv2.cv2Color(img, cv2.COLOR_RGBA2BGRA)
    if (out_img.dtype == np.float32 or out_img.dtype == np.float64) and not is_hdr:
        out_img = np.clip(out_img, 0, 1) * 255
        out_img = out_img.astype(np.uint8)
    if is_hdr:
        out_img = out_img.astype(np.float32)
        
    cv2.imwrite(path, out_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    

#==============================================

# Check if image format should be one of hdr_extensions
def is_hdr_from_file_extension(file_path):
    extension = os.path.splitext(file_path)[1]
    return extension in hdr_extensions

#==============================================

# get spatial resolution of an image
def get_image_resolution(img):
    return img.shape[:2][::-1]


def create_video(dir, outpath, fps=30, bitrate="12M"):
    
    print("Collecting files...")

    out_file = os.path.join(dir, "sequence.mp4")
    img_files = find_images(dir)

    video_writer = imageio.get_writer(out_file, mode='I', fps=fps, codec='libx264', bitrate=bitrate)
    
    print("Compiling video...")

    def add_frame(img_file):
        img = load_image(img_file, normalize=False)
        if img.shape[0] <= 32:
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        video_writer.append_data(img)
    
    for img_file in img_files:
        add_frame(img_file)

    video_writer.close()

    print("Done.")



def create_video_tiled_gt(dir1, dir2, gt, outpath, fps=10, bitrate="12M"):
    
    print("Collecting files...")

    out_file = outpath
    img_files = find_images(dir1)
    img_files1 = find_images(dir2)

    video_writer = imageio.get_writer(out_file, mode='I', fps=fps, codec='libx264', bitrate=bitrate)
    
    print("Compiling video...")
    gt = np.clip(gt, 0, 1) * 255
    gt = gt.astype(np.uint8)

    def add_frame(img_file1, img_file2, gt):
        img1 = load_image(img_file1, normalize=False)
        if img1.shape[0] == 32:
            img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
        
        
        img2 = load_image(img_file2, normalize=False)
        if img2.shape[0] == 32:
            img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_AREA)
        
        img = np.concatenate((img1, img2, gt), axis=1)

        video_writer.append_data(img)
    
    for i in range(len(img_files)):
        add_frame(img_files[i], img_files1[i], gt)

    video_writer.close()

    print("Done.")
