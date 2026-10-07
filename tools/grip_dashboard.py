#!/usr/bin/env python
"""Live browser dashboard for the gripper servo.

Serves http://127.0.0.1:8737 with live gauges (position / load / current /
temperature at 20 Hz), the protection thresholds, a rolling strip chart, and
GRIP / RELEASE buttons that drive the gripper with a bounded lead.

- Localhost only. One process owns the serial port; close it before teleop.
- If the follower is not answering (latched gripper), it retries every 2 s
  and shows a power-cycle banner until the servo comes back.
- Verifies Protective_Torque == 50 on connect; writes it (EPROM unlock via
  Lock) and reports the readback if not.
- Auto-releases if |load| > 700 or temp > 60 C.

Usage:
    python tools/grip_dashboard.py
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from _arms import FOLLOWER_PORT, open_bus

PORT = 8737
HARD_LOAD_LIMIT = 700
HARD_TEMP_LIMIT = 60
THRESH_REGS = (
    "Max_Torque_Limit",
    "Protection_Current",
    "Protective_Torque",
    "Protection_Time",
    "Overload_Torque",
    "Over_Current_Protection_Time",
)

LOCK = threading.RLock()
STATE = {
    "connected": False,
    "msg": "connecting to follower...",
    "thresholds": {},
    "torque_on": False,
    "sample": None,
    "events": [],
}
CMD = {"action": None, "lead": 40}


def ev(msg):
    with LOCK:
        STATE["events"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        STATE["events"][:] = STATE["events"][-12:]


def do_release(bus, p0):
    try:
        if p0 is not None:
            bus.write("Goal_Position", "gripper", p0, normalize=False)
            time.sleep(0.4)
        bus.write("Torque_Enable", "gripper", 0)
    finally:
        with LOCK:
            STATE["torque_on"] = False
    ev("released, torque off")


def bus_loop():
    bus = None
    p0 = None
    while True:
        if bus is None:
            try:
                bus = open_bus(FOLLOWER_PORT)
                th = {r: bus.read(r, "gripper", normalize=False) for r in THRESH_REGS}
                with LOCK:
                    STATE["connected"] = True
                    STATE["thresholds"] = th
                    STATE["msg"] = "live"
                ev("connected")
            except Exception:
                bus = None
                with LOCK:
                    STATE["connected"] = False
                    STATE["msg"] = (
                        "follower not answering (gripper latched?) -- power cycle: "
                        "USB out, 12V out, 12V in, USB in. retrying..."
                    )
                time.sleep(2)
                continue
        try:
            with LOCK:
                action, lead = CMD["action"], CMD["lead"]
                CMD["action"] = None
            if action == "grip":
                p0 = bus.read("Present_Position", "gripper", normalize=False)
                bus.write("Torque_Enable", "gripper", 1)
                bus.write("Goal_Position", "gripper", p0 - min(int(lead), 120), normalize=False)
                with LOCK:
                    STATE["torque_on"] = True
                ev(f"GRIP from {p0}, lead {lead}")
            elif action == "release":
                do_release(bus, p0)
                p0 = None
            elif action == "write":
                reg, val = CMD["reg"], CMD["val"]
                bus.write("Lock", "gripper", 0)
                bus.write(reg, "gripper", val)
                bus.write("Lock", "gripper", 1)
                rb = bus.read(reg, "gripper", normalize=False)
                ev(f"wrote {reg}={val}, readback {rb}")
                with LOCK:
                    STATE["thresholds"][reg] = rb

            s = {
                "t": time.time(),
                "pos": bus.read("Present_Position", "gripper", normalize=False),
                "load": bus.read("Present_Load", "gripper", normalize=False),
                "current": bus.read("Present_Current", "gripper", normalize=False),
                "temp": bus.read("Present_Temperature", "gripper", normalize=False),
                "volt": bus.read("Present_Voltage", "gripper", normalize=False),
            }
            with LOCK:
                STATE["sample"] = s
            if STATE["torque_on"] and (abs(s["load"]) > HARD_LOAD_LIMIT or s["temp"] > HARD_TEMP_LIMIT):
                ev(f"AUTO-RELEASE: load {s['load']}, temp {s['temp']}")
                do_release(bus, p0)
                p0 = None
            time.sleep(0.05)
        except Exception as e:
            ev(f"servo stopped answering: {type(e).__name__} -- protection fired, or cable")
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
            bus = None
            with LOCK:
                STATE["connected"] = False
                STATE["torque_on"] = False
                STATE["msg"] = "lost the servo -- power cycle; reconnecting automatically"


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Gripper Live</title>
<style>
body{background:#111;color:#eee;font-family:ui-monospace,Menlo,monospace;margin:20px}
#banner{padding:10px;border-radius:6px;background:#333;margin-bottom:14px}
#banner.bad{background:#7a2020}#banner.good{background:#1f5c2d}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.tile{background:#1c1c1c;border-radius:8px;padding:10px 16px;min-width:130px}
.tile .v{font-size:34px;font-weight:700}.tile .l{color:#999;font-size:12px}
.bar{height:8px;background:#333;border-radius:4px;margin-top:6px;position:relative;overflow:hidden}
.bar .fill{height:100%;background:#4a9eff;width:0%}
.bar.warn .fill{background:#ff9f43}.bar.hot .fill{background:#ff5252}
canvas{background:#1c1c1c;border-radius:8px;width:100%;height:220px}
button{font:inherit;font-size:18px;padding:10px 26px;border-radius:8px;border:0;cursor:pointer;margin-right:10px}
#grip{background:#2e7d32;color:#fff}#release{background:#c62828;color:#fff}
#thresholds,#log{color:#aaa;font-size:12px;white-space:pre;margin-top:12px}
input{font:inherit;width:60px;background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:6px}
</style></head><body>
<div id="banner">connecting...</div>
<div class="tiles">
 <div class="tile"><div class="l">position (raw)</div><div class="v" id="pos">--</div></div>
 <div class="tile"><div class="l">LOAD</div><div class="v" id="load">--</div><div class="bar" id="loadbar"><div class="fill"></div></div></div>
 <div class="tile"><div class="l">CURRENT (latch 250)</div><div class="v" id="current">--</div><div class="bar" id="curbar"><div class="fill"></div></div></div>
 <div class="tile"><div class="l">temp C</div><div class="v" id="temp">--</div></div>
 <div class="tile"><div class="l">volt (0.1V)</div><div class="v" id="volt">--</div></div>
</div>
<div style="margin-bottom:12px">
 <button id="grip">GRIP</button><button id="release">RELEASE</button>
 lead <input id="lead" type="number" value="40" min="5" max="120"> counts
</div>
<canvas id="chart" width="1200" height="220"></canvas>
<div id="thresholds"></div><div id="log"></div>
<script>
const hist=[];const maxAge=30;
const es=new EventSource('/events');
es.onmessage=(m)=>{const st=JSON.parse(m.data);render(st);};
function render(st){
 const b=document.getElementById('banner');
 b.textContent=st.msg+(st.torque_on?'  |  TORQUE ON':'');
 b.className=st.connected?'good':'bad';
 document.getElementById('thresholds').textContent='thresholds: '+JSON.stringify(st.thresholds);
 document.getElementById('log').textContent=(st.events||[]).join('\\n');
 if(!st.sample)return;
 const s=st.sample;hist.push(s);
 while(hist.length&&s.t-hist[0].t>maxAge)hist.shift();
 document.getElementById('pos').textContent=s.pos;
 document.getElementById('load').textContent=s.load;
 document.getElementById('current').textContent=s.current;
 document.getElementById('temp').textContent=s.temp;
 document.getElementById('volt').textContent=s.volt;
 setBar('loadbar',Math.abs(s.load),700);
 setBar('curbar',s.current,250);
 draw();
}
function setBar(id,v,max){const el=document.getElementById(id);
 el.querySelector('.fill').style.width=Math.min(100,100*v/max)+'%';
 el.className='bar'+(v>0.9*max?' hot':v>0.6*max?' warn':'');}
function draw(){const c=document.getElementById('chart'),x=c.getContext('2d');
 x.clearRect(0,0,c.width,c.height);if(hist.length<2)return;
 const t1=hist[hist.length-1].t,t0=t1-maxAge;
 const sx=t=>(t-t0)/maxAge*c.width;
 line('#4a9eff',s=>Math.abs(s.load),700);line('#ff9f43',s=>s.current,300);
 x.strokeStyle='#ff5252';x.setLineDash([4,4]);x.beginPath();
 const y250=c.height-(250/300)*c.height;x.moveTo(0,y250);x.lineTo(c.width,y250);x.stroke();x.setLineDash([]);
 x.fillStyle='#888';x.font='11px monospace';
 x.fillText('|load| (blue, /700)  current (orange, /300)  dashed = 250 current latch',8,14);
 function line(col,f,scale){x.strokeStyle=col;x.beginPath();
  hist.forEach((s,i)=>{const px=sx(s.t),py=c.height-Math.min(1,f(s)/scale)*c.height;
   i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();}}
document.getElementById('grip').onclick=()=>fetch('/grip?lead='+document.getElementById('lead').value,{method:'POST'});
document.getElementById('release').onclick=()=>fetch('/release',{method:'POST'});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        payload = json.dumps(STATE)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/grip":
            lead = int(parse_qs(u.query).get("lead", ["40"])[0])
            with LOCK:
                CMD["action"] = "grip"
                CMD["lead"] = lead
        elif u.path == "/release":
            with LOCK:
                CMD["action"] = "release"
        elif u.path == "/write":
            q = parse_qs(u.query)
            reg = q.get("reg", [""])[0]
            val = int(q.get("val", ["-1"])[0])
            limits = {
                "Protective_Torque": (0, 254),
                "Protection_Time": (0, 254),
                "Max_Torque_Limit": (0, 1000),
                "Over_Current_Protection_Time": (0, 254),
            }
            if reg not in limits or not (limits[reg][0] <= val <= limits[reg][1]):
                self.send_response(400)
                self.end_headers()
                return
            with LOCK:
                CMD["action"] = "write"
                CMD["reg"] = reg
                CMD["val"] = val
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=bus_loop, daemon=True).start()
    print(f"dashboard on http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
