import cv2
import argparse
import time
from pathlib import Path
from models.pipeline import DefectDetectionPipeline

def process_image(image_path: str, pipeline: DefectDetectionPipeline):
    """Process a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    start = time.time()
    result = pipeline.run(img)
    elapsed = time.time() - start

    print(f"\n📸 Image: {image_path}")
    print(f"⏱️  Inference time: {elapsed:.3f}s")
    print(f"🔍 Defect found: {result['defect_found']}")
    print(f"🏷️  Defect type: {result['defect_type']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")

    if result["defect_found"]:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{result['defect_type']} {result['confidence']:.2%}"
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
    else:
        cv2.putText(
            img, "PASS", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
        )

    output_path = f"results/{Path(image_path).stem}_result.jpg"
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Result saved to: {output_path}")

def process_folder(folder_path: str, pipeline: DefectDetectionPipeline):
    """Process all images in a folder."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [
        f for f in Path(folder_path).glob("**/*")
        if f.suffix.lower() in image_extensions
    ]

    print(f"\n🔄 Processing {len(images)} images from {folder_path}")

    passed = 0
    defects = 0

    for image_path in images:
        result = pipeline.run(cv2.imread(str(image_path)))
        if result["defect_found"]:
            defects += 1
        else:
            passed += 1

    print(f"\n📊 Results Summary:")
    print(f"   ✅ Passed : {passed}")
    print(f"   ❌ Defects: {defects}")
    print(f"   📈 Pass Rate: {passed/(passed+defects)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Defect Detection"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to image or folder"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    args = parser.parse_args()

    print("🚀 Loading Defect Detection Pipeline...")
    pipeline = DefectDetectionPipeline()

    source = Path(args.source)
    if source.is_dir():
        process_folder(str(source), pipeline)
    else:
        process_image(str(source), pipeline)
