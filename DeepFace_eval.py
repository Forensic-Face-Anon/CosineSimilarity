import os
import cv2
import csv
from deepface import DeepFace

# ---------------- CONFIG ----------------
# Change these filenames accordingly and on line 56!

ORIGINAL_DIR = "./driving/"
ANON_DIR = "./output_mask/" 
OUTPUT_CSV = "./identity_verification_results.csv"

FRAME_POSITIONS = ["first", "middle", "last"]
# ----------------------------------------


def extract_frame(video_path, position):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    if position == "first":
        frame_id = 0
    elif position == "middle":
        frame_id = total_frames // 2
    elif position == "last":
        frame_id = total_frames - 1
    else:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def save_temp_frame(frame, path):
    cv2.imwrite(path, frame)


results = []

for vid in sorted(os.listdir(ORIGINAL_DIR)):
    if not vid.endswith(".mp4"):
        continue

    orig_path = os.path.join(ORIGINAL_DIR, vid)
    anon_name = "mask_iris_" + vid  # "flow_" + vid
    anon_path = os.path.join(ANON_DIR, anon_name)

    if not os.path.exists(anon_path):
        print(f"[SKIP] Missing anonymised video for {vid}")
        continue

    similarities = []
    verified_flags = []

    for pos in FRAME_POSITIONS:
        orig_frame = extract_frame(orig_path, pos)
        anon_frame = extract_frame(anon_path, pos)

        if orig_frame is None or anon_frame is None:
            similarities.append("NA")
            verified_flags.append(False)
            continue

        orig_img = f"tmp_orig_{pos}.jpg"
        anon_img = f"tmp_anon_{pos}.jpg"

        save_temp_frame(orig_frame, orig_img)
        save_temp_frame(anon_frame, anon_img)

        try:
            result = DeepFace.verify(
                img1_path=orig_img,
                img2_path=anon_img,
                enforce_detection=False
            )
            similarities.append(result.get("distance", "NA"))
            verified_flags.append(result.get("verified", False))
        except Exception as e:
            similarities.append("NA")
            verified_flags.append(False)

        os.remove(orig_img)
        os.remove(anon_img)

    # ✅ Overall verdict logic (conservative)
    overall_verdict = any(verified_flags)

    results.append([
        vid,
        similarities[0],
        similarities[1],
        similarities[2],
        overall_verdict
    ])


# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "cosine_similarity_first",
        "cosine_similarity_middle",
        "cosine_similarity_last",
        "identity_recoverable"
    ])
    writer.writerows(results)

print("✅ Identity verification results saved to:", OUTPUT_CSV)
