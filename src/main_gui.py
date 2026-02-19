# main_gui.py
import threading
from ui_bus import EventBus
from ui_ctk import PacketVisionApp
from main import run_workflow
from main import run_workflow
# change to your filename

def start_worker(bus, stop_event):
    t = threading.Thread(target=run_workflow, args=(bus, stop_event), daemon=True)
    t.start()

if __name__ == "__main__":
    bus = EventBus()
    app = PacketVisionApp(bus, start_worker)
    app.mainloop()
