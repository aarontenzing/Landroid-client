import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Create the pipeline
pipeline = Gst.parse_launch("gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H265\
 ! rtph265depay ! avdec_h265 ! videoscale ! videoconvert ! ximagesink")

# Start playing
pipeline.set_state(Gst.State.PLAYING)

# Run until the pipeline is stopped or an error occurs
try:
    bus = pipeline.get_bus()
    while True:
        message = bus.timed_pop_filtered(10000, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"Error: {err}, {debug}")
                break
            elif message.type == Gst.MessageType.EOS:
                print("End of stream")
                break
finally:
    # Cleanup
    pipeline.set_state(Gst.State.NULL)
