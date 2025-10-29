#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detection import main as run_detection, register_di_callback, register_di_callback2

from indy7 import indyCTL

indy = indyCTL(ip="192.168.0.6")

def on_di(di: dict):
    print(
        "[Ripe XYZ] "
        f"X={di.get('X', float('nan')):.3f} m, "
        f"Y={di.get('Y', float('nan')):.3f} m, "
        f"Z={di.get('Z', float('nan')):.3f} m | "
        f"dist={di.get('distance_m', float('nan')):.3f} m, "
        f"yaw={di.get('yaw_deg', float('nan')):.2f}°, "
        f"pitch={di.get('pitch_deg', float('nan')):.2f}°, "
        f"theta={di.get('theta_deg', float('nan')):.2f}°"
    )
    indy.set_point(cam_x=di.get("X"), depth=di.get("Z"))

def on_di2(di: dict, angle):
    print(
        "[Ripe XYZ] "
        f"X={di.get('X', float('nan')):.3f} m, "
        f"Y={di.get('Y', float('nan')):.3f} m, "
        f"Z={di.get('Z', float('nan')):.3f} m | "
        f"dist={di.get('distance_m', float('nan')):.3f} m, "
        f"yaw={di.get('yaw_deg', float('nan')):.2f}°, "
        f"pitch={di.get('pitch_deg', float('nan')):.2f}°, "
        f"theta={di.get('theta_deg', float('nan')):.2f}°"
    )
    indy.run(cam_x=di.get("X"), cam_y=di.get("Y"), cam_z=di.get("Z"), angle=angle)
    # indy.run(cam_x=di.get("X"), cam_y=di.get("Y"), cam_z=0.2, angle=0)

if __name__ == "__main__":
    register_di_callback(on_di)
    register_di_callback2(on_di2)
    run_detection()
    indy.close()











# if key_m_flag == True:
#     if count_flag == 1:
#         indy.set_point(cam_x=di.get("X"), depth=di.get("Z"))
#         count_flag = 2
#     elif count_flag == 2:
#         indy.run(cam_x=di.get("X"), cam_y=di.get("Y"), cam_z=di.get("Z"), angle=0)
#         indy.run(cam_x=di.get("X"), cam_y=di.get("Y"), cam_z=0.2, angle=0)
#         count_flag = 1
#         key_m_flag = False
# else:
#     if key == ord('q'):
#         break
#     elif key == ord('1'):
#         if _LAST_DI is not None:
#             if _DI_CB is not None:
#                 try:
#                     _DI_CB(_LAST_DI)
#                 except Exception as e:
#                     print(f"[WARN] DI callback error: {e}")
#             else:
#                 d = _LAST_DI
#                 print(f"[Ripe XYZ] X={d.get('X', float('nan')):.3f} m, "
#                     f"Y={d.get('Y', float('nan')):.3f} m, "
#                     f"Z={d.get('Z', float('nan')):.3f} m "
#                     f"(dist={d.get('distance_m', float('nan')):.3f} m)")
#         else:
#             print("[INFO] 아직 Ripe 포인트가 감지되지 않았어.")
#     elif key == ord('m'):
#         key_m_flag = True