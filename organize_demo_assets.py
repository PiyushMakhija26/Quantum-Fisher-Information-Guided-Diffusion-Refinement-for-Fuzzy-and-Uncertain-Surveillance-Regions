import os
import shutil
import cv2
import numpy as np

def create_comparison_figure():
    assets_dir = os.path.join("assets", "demo")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Copy paired inputs and restored outputs
    pairs = [
        ("data/demo/sample_dark_01.png", "outputs/demo/sample_dark_01.png", "outputs/demo/qfi_mask_sample_dark_01.png", "01"),
        ("data/demo/sample_dark_02.png", "outputs/demo/sample_dark_02.png", "outputs/demo/qfi_mask_sample_dark_02.png", "02"),
        ("data/demo/sample_dark_03.png", "outputs/demo/sample_dark_03.png", "outputs/demo/qfi_mask_sample_dark_03.png", "03"),
    ]
    
    for in_path, out_path, mask_path, idx in pairs:
        shutil.copy(in_path, os.path.join(assets_dir, f"input_{idx}.png"))
        shutil.copy(out_path, os.path.join(assets_dir, f"restored_{idx}.png"))
        print(f"Copied input_{idx}.png and restored_{idx}.png to {assets_dir}")

    # Build qualitative comparison montage
    # Grid: 2 rows (sample 1, sample 2), 3 columns: [Degraded Input] -> [QFI Reliability Mask] -> [QFI-Diff Restored Output]
    header_h = 50
    col_w = 512
    row_h = 512
    gap = 20
    
    total_w = 3 * col_w + 4 * gap
    total_h = header_h + 2 * row_h + 3 * gap
    
    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)  # Clean light gray background
    
    # Column headers
    titles = [
        "Degraded Surveillance Input",
        "QFI Reliability Mask (2D TFIM)",
        "QFI-Diff Restored Output"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c_idx, title in enumerate(titles):
        col_x = gap + c_idx * (col_w + gap)
        text_size = cv2.getTextSize(title, font, 0.85, 2)[0]
        text_x = col_x + (col_w - text_size[0]) // 2
        text_y = (header_h + text_size[1]) // 2 + 5
        cv2.putText(canvas, title, (text_x, text_y), font, 0.85, (30, 30, 30), 2, cv2.LINE_AA)

    # Place rows
    for r_idx in range(2):
        in_path, out_path, mask_path, idx = pairs[r_idx]
        y_pos = header_h + gap + r_idx * (row_h + gap)
        
        img_in = cv2.imread(in_path)
        img_mask = cv2.imread(mask_path)
        img_out = cv2.imread(out_path)
        
        # Colorize mask with COLORMAP_INFERNO or JET for striking visual impact
        mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        mask_color = cv2.applyColorMap(mask_gray, cv2.COLORMAP_INFERNO)
        
        images = [img_in, mask_color, img_out]
        for c_idx, img in enumerate(images):
            col_x = gap + c_idx * (col_w + gap)
            # Add a subtle border
            cv2.rectangle(canvas, (col_x - 2, y_pos - 2), (col_x + col_w + 1, y_pos + row_h + 1), (200, 200, 200), 2)
            canvas[y_pos : y_pos + row_h, col_x : col_x + col_w] = img

    comp_path = os.path.join(assets_dir, "qualitative_comparison.png")
    cv2.imwrite(comp_path, canvas)
    print(f"Generated qualitative comparison figure at {comp_path} (size: {canvas.shape})")

if __name__ == "__main__":
    create_comparison_figure()
