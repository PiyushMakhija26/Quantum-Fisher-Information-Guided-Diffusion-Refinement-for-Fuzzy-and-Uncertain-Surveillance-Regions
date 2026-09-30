import os
import subprocess
import argparse

def evaluate_yolov5(restored_dir, labels_dir):
    if not os.path.exists("yolov5"):
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    
    yaml_content = f"""
    val: {restored_dir}
    nc: 1
    names: ['face']
    """
    with open("yolov5/darkface_eval.yaml", "w") as f:
        f.write(yaml_content)

    val_command = [
        "python", "yolov5/val.py",
        "--weights", "yolov5s.pt",
        "--data", "yolov5/darkface_eval.yaml",
        "--task", "val",
        "--conf-thres", "0.25",
        "--iou-thres", "0.50"
    ]
    
    subprocess.run(val_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    evaluate_yolov5(parser.parse_args().pred_dir, parser.parse_args().labels_dir)
