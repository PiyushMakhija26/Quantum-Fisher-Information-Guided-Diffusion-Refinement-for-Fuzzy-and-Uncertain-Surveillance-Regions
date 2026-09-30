import os
import cv2
import numpy as np

def generate_surveillance_degradation(img_clean, seed=42):
    np.random.seed(seed)
    h, w, c = img_clean.shape
    
    # 1. Non-uniform low-light illumination attenuation field
    # Create smooth vignette / directional shadow gradient
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    center_x = np.random.uniform(-0.3, 0.3)
    center_y = np.random.uniform(-0.3, 0.3)
    dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    illumination = np.exp(-dist * 1.8) * 0.25 + 0.05
    
    # Additional horizontal gradient simulating off-axis street lamp / shadow
    grad = np.linspace(0.8, 0.2, w).reshape(1, w, 1)
    illumination = np.clip(illumination[:, :, np.newaxis] * grad, 0.02, 0.4)
    
    img_dark = (img_clean / 255.0) * illumination
    
    # 2. Realistic Poisson (photon) noise + Gaussian (electronic readout) sensor noise
    peak = 50.0  # Photon flux scaling
    poisson_noisy = np.random.poisson(np.clip(img_dark * peak, 0, None)) / peak
    readout_noise = np.random.normal(0, 0.03, (h, w, c))
    
    degraded = np.clip(poisson_noisy + readout_noise, 0.0, 1.0) * 255.0
    return degraded.astype(np.uint8)

def create_synthetic_surveillance_scene(scene_id=1):
    np.random.seed(scene_id * 100)
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Background architectural texture (walls / corridor)
    for i in range(0, 512, 64):
        cv2.line(canvas, (0, i), (512, i), (70, 70, 75), 2)
        cv2.line(canvas, (i, 0), (i, 512), (65, 65, 70), 2)
        
    # Foreground subjects / facial silhouettes
    if scene_id == 1:
        # Subject head & shoulders
        cv2.ellipse(canvas, (256, 380), (120, 150), 0, 0, 360, (140, 120, 110), -1)
        cv2.circle(canvas, (256, 220), 80, (190, 160, 140), -1)
        # Facial contours
        cv2.circle(canvas, (230, 205), 10, (80, 50, 40), -1)
        cv2.circle(canvas, (282, 205), 10, (80, 50, 40), -1)
        cv2.ellipse(canvas, (256, 235), (8, 16), 0, 0, 360, (150, 120, 100), -1)
        cv2.ellipse(canvas, (256, 260), (25, 10), 0, 0, 360, (120, 70, 60), -1)
    elif scene_id == 2:
        # Perimeter subject at distance
        cv2.rectangle(canvas, (100, 150), (412, 450), (100, 100, 110), -1)
        cv2.circle(canvas, (256, 200), 65, (180, 150, 130), -1)
        cv2.circle(canvas, (235, 190), 8, (60, 40, 30), -1)
        cv2.circle(canvas, (277, 190), 8, (60, 40, 30), -1)
        cv2.ellipse(canvas, (256, 230), (20, 8), 0, 0, 360, (110, 60, 50), -1)
    else:
        # Off-center subject with high-contrast background elements
        cv2.circle(canvas, (210, 240), 90, (195, 165, 145), -1)
        cv2.ellipse(canvas, (210, 420), (140, 160), 0, 0, 360, (130, 110, 100), -1)
        cv2.circle(canvas, (180, 225), 12, (70, 45, 35), -1)
        cv2.circle(canvas, (240, 225), 12, (70, 45, 35), -1)
        cv2.ellipse(canvas, (210, 275), (28, 12), 0, 0, 360, (130, 80, 70), -1)
        # Background sign/window
        cv2.rectangle(canvas, (360, 80), (480, 220), (200, 200, 220), -1)

    return canvas

def main():
    out_dir = os.path.join("data", "demo")
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(1, 4):
        clean_img = create_synthetic_surveillance_scene(i)
        degraded_img = generate_surveillance_degradation(clean_img, seed=42 + i)
        
        filename = f"sample_dark_0{i}.png"
        filepath = os.path.join(out_dir, filename)
        cv2.imwrite(filepath, degraded_img)
        print(f"Generated {filepath} (shape: {degraded_img.shape}, dtype: {degraded_img.dtype})")

if __name__ == "__main__":
    main()
