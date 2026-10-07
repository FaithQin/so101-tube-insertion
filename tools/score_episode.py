"""Score a trial from the episode's OWN recorded video, not from a live grab.

Why this exists (Aug 29 2026): `seat_check.py` opens the camera *after* the
rollout process exits, which is after the episode's reset phase. It produced
two wrong verdicts in one block:

  * It photographed a scene the operator had already reset, so the frame had
    nothing to do with the state at rollout end.
  * It asks only "is the blue cap inside the rack region", so a cap held in the
    GRIPPER directly above the funnel scored as SEAT. On the discarded trial 3
    those false seats lasted 0.1 s and 0.5 s while the arm carried the tube.

Success is defined by the protocol as "tube standing in the target hole
unsupported at rollout end", and the recording is the only evidence
synchronized to that moment. Reporting the seat RUNS alongside the final
verdict is what separates "seated and held" from "cap passed over the funnel".

It also enforces the start-state guard. That same trial BEGAN with the tube
already seated and was reset ~7 s into recording with a hand in frame, so the
policy observed 7 s of a scene that exists nowhere in training. Nothing in the
harness caught it — the operator did. An episode that opens seated is INFRA and
carries no verdict at all, so it cannot slip into a block.

The classifier is deliberately identical to seat_check.py (blue-cap HSV mask,
top 250 rows zeroed, largest contour > 40 px, seat box 470<x<640, 400<y<540),
which was validated on all 46 v3 episodes. tests/test_episode_scoring.py pins
the two together; if seat_check.py's region changes, that test fails.

    python tools/score_episode.py <episode.mp4>
"""

import sys

import cv2
import numpy as np

SEAT_X = (470, 640)
SEAT_Y = (400, 540)
MASK_TOP_ROWS = 250
MIN_CONTOUR_AREA = 40
HSV_LO = np.array([100, 120, 60])
HSV_HI = np.array([125, 255, 255])

# An episode whose opening seconds show the cap already in the rack never had a
# valid start state. 2.0 s is generous: the arm cannot reach the rack that fast.
INFRA_START_WINDOW_S = 2.0


def classify_frame(img):
    """Return (cap_centroid | None, is_in_seat_region)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask[:MASK_TOP_ROWS, :] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    if not contours:
        return None, False
    m = cv2.moments(max(contours, key=cv2.contourArea))
    if m["m00"] == 0:
        return None, False
    x, y = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    seat = SEAT_X[0] < x < SEAT_X[1] and SEAT_Y[0] < y < SEAT_Y[1]
    return (x, y), seat


def score_frames(frames, fps: float = 20.0) -> dict:
    """Score an iterable of BGR frames as one episode."""
    seat_runs, run_start = [], None
    last_cap, last_seat = None, False
    n = 0

    for i, img in enumerate(frames):
        cap, seat = classify_frame(img)
        if cap is not None:
            last_cap = cap
        last_seat = seat
        if seat and run_start is None:
            run_start = i
        elif not seat and run_start is not None:
            seat_runs.append((run_start, i - 1))
            run_start = None
        n = i + 1

    if run_start is not None:
        seat_runs.append((run_start, n - 1))

    runs = [(a / fps, b / fps, (b - a + 1) / fps) for a, b in seat_runs]
    infra = any(a <= INFRA_START_WINDOW_S for a, _, _ in runs)

    if last_cap is None:
        final = "UNKNOWN"
    else:
        final = "SEAT" if last_seat else "MISS"

    return {
        "n": n,
        "duration_s": n / fps if fps else 0.0,
        "final": final,
        "final_cap": last_cap,
        "seat_runs": runs,
        "held_to_end": bool(last_seat),
        "infra_start_seated": infra,
        "scoreable": not infra,
    }


def score_video(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    def _frames():
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img

    try:
        return score_frames(_frames(), fps=fps)
    finally:
        cap.release()


def render(r: dict) -> str:
    lines = [f"  frames={r['n']} dur={r['duration_s']:.1f}s"]
    if r["infra_start_seated"]:
        lines.append("  *** INFRA: episode BEGAN with the cap in the rack region — invalid start state, NOT scoreable ***")
    lines.append(f"  FINAL: cap={r['final_cap']} -> {r['final']}")
    for a, b, d in r["seat_runs"]:
        tag = "holds to end" if abs(b - (r["duration_s"] - 1 / 20)) < 0.2 else "LOST"
        lines.append(f"  seated {a:5.1f}s -> {b:5.1f}s ({d:.1f}s){'  [' + tag + ']'}")
    if not r["seat_runs"]:
        lines.append("  never seated at any point")
    return "\n".join(lines)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        print(render(score_video(p)))
