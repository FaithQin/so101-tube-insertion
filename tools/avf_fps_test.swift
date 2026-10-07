// Ground-truth camera fps test using native AVFoundation — no OpenCV, no ffmpeg.
// Locks each icspring camera to its 30 fps format explicitly, streams 5 s,
// reports delivered fps AND OS-dropped frames. If this hits 30, the 20 fps
// problem lives in OpenCV's capture layer; if this also sits at 20, it's the
// OS/driver and no Python-side fix exists.
//
// Run in a TCC-granted terminal:  swift tools/avf_fps_test.swift

import AVFoundation
import Foundation

final class Counter: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var frames = 0
    var dropped = 0
    func captureOutput(_ o: AVCaptureOutput, didOutput s: CMSampleBuffer, from c: AVCaptureConnection) {
        frames += 1
    }
    func captureOutput(_ o: AVCaptureOutput, didDrop s: CMSampleBuffer, from c: AVCaptureConnection) {
        dropped += 1
    }
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external, .builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
)

for device in discovery.devices {
    print("\n=== \(device.localizedName) (\(device.uniqueID)) ===")
    guard device.localizedName.lowercased().contains("icspring") else {
        print("  (skipping — not a rig camera)")
        continue
    }

    // Pick the format: 1280x720 if offered (front), else 640x480 (wrist).
    var chosen: AVCaptureDevice.Format?
    for want in [(1280, 720), (640, 480)] {
        for f in device.formats {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let rates = f.videoSupportedFrameRateRanges.map { $0.maxFrameRate }
            if Int(d.width) == want.0 && Int(d.height) == want.1 && rates.contains(where: { $0 >= 29 }) {
                chosen = f
                break
            }
        }
        if chosen != nil { break }
    }
    guard let format = chosen else {
        print("  no 30fps format at 1280x720 or 640x480 — formats:")
        for f in device.formats {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let rates = f.videoSupportedFrameRateRanges.map { "\($0.minFrameRate)-\($0.maxFrameRate)" }
            print("    \(d.width)x\(d.height) @ \(rates)")
        }
        continue
    }

    let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
    let fourcc = CMFormatDescriptionGetMediaSubType(format.formatDescription)
    let fourccStr = String(bytes: [
        UInt8((fourcc >> 24) & 255), UInt8((fourcc >> 16) & 255),
        UInt8((fourcc >> 8) & 255), UInt8(fourcc & 255),
    ], encoding: .ascii) ?? "????"

    do {
        let session = AVCaptureSession()
        let input = try AVCaptureDeviceInput(device: device)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = false
        let counter = Counter()
        output.setSampleBufferDelegate(counter, queue: DispatchQueue(label: "cam"))

        session.beginConfiguration()
        session.addInput(input)
        session.addOutput(output)
        try device.lockForConfiguration()
        device.activeFormat = format
        // The driver accepts ONLY durations from the format's own ranges
        // (e.g. exactly 1000000/30000030): a hand-built 1/30 throws NSInvalidArgumentException.
        if let range = format.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) {
            device.activeVideoMinFrameDuration = range.minFrameDuration
            device.activeVideoMaxFrameDuration = range.minFrameDuration
        }
        device.unlockForConfiguration()
        session.commitConfiguration()

        session.startRunning()

        // Verify nothing reset our choice when the session started.
        let active = device.activeFormat
        let ad = CMVideoFormatDescriptionGetDimensions(active.formatDescription)
        print("  active after start: \(ad.width)x\(ad.height), "
            + "min duration \(device.activeVideoMinFrameDuration.value)/\(device.activeVideoMinFrameDuration.timescale)")
        Thread.sleep(forTimeInterval: 1.0) // warmup
        counter.frames = 0
        counter.dropped = 0
        let t0 = Date()
        Thread.sleep(forTimeInterval: 5.0)
        let dt = -t0.timeIntervalSinceNow
        session.stopRunning()

        let minDur = device.activeVideoMinFrameDuration
        print("  format: \(dims.width)x\(dims.height) \(fourccStr), locked to 30 fps "
            + "(active min duration \(minDur.value)/\(minDur.timescale))")
        print(String(format: "  delivered %.1f fps, OS-dropped %d frames in %.1f s",
                     Double(counter.frames) / dt, counter.dropped, dt))
    } catch {
        print("  ERROR: \(error)")
    }
}
