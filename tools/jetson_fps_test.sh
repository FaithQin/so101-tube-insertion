#!/usr/bin/env bash
# Jetson USB-camera delivered-fps test — the "free test" from the Aug 16 handoff.
#
# QUESTION IT ANSWERS: is the 20.0/22.7-instead-of-30 fps decimation a camera
# problem or a macOS problem? On the Mac the capture stack throttles both
# icspring cameras by deterministic ratios (2/3 and 3/4). If the SAME cameras
# deliver ~30.0 here via V4L2, the cameras are innocent, macOS is guilty, and
# a 30 fps recording rig exists without buying anything: this board.
#
# Run ON the Jetson (JetPack 6.2), with both icspring cameras plugged into the
# USB-A 3.0 ports:
#     sudo apt-get install -y v4l-utils   # usually preinstalled
#     bash jetson_fps_test.sh
#
# READING THE OUTPUT: the number that matters is the "fps:" v4l2-ctl prints
# during streaming — that is DELIVERED frames, kernel-level, no OpenCV, no
# driver readback lies (the Mac lesson: measure delivery, never trust claims).
# Pass bar, matching the Mac recording configs: >=29.5 at 1280x720 and 640x480.

set -u

echo "=== USB video devices ==="
v4l2-ctl --list-devices 2>/dev/null

# Each USB camera typically exposes two /dev/video* nodes; only the first of
# each pair streams. Probe every node and skip the ones that refuse.
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    echo ""
    echo "=== $dev: supported formats (abridged) ==="
    v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null | grep -E "\[[0-9]\]|Size|Interval.*(30\.000|60\.000|120\.000)" | head -25

    for cfg in "1280,720" "640,480"; do
        w=${cfg%,*}; h=${cfg#*,}
        for fmt in MJPG YUYV; do
            echo ""
            echo "--- $dev  ${w}x${h} $fmt, requesting 30 fps, 5 s stream ---"
            v4l2-ctl -d "$dev" \
                --set-fmt-video=width=$w,height=$h,pixelformat=$fmt \
                --set-parm=30 \
                --stream-mmap --stream-count=150 --stream-to=/dev/null 2>&1 \
                | tail -2
        done
    done
done

echo ""
echo "Done. 'fps: 30.0x' at 1280x720 on one node per camera = cameras innocent,"
echo "macOS guilty, and this board can be the 30 fps recording rig as-is."
