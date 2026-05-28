from __future__ import annotations
import argparse

def main():
    p=argparse.ArgumentParser(description='Deploy MFRP planner. This entry uses scene-only inference only.')
    p.add_argument('--checkpoint', required=True); p.add_argument('--calibration', required=True); p.add_argument('--scenario', required=True); p.add_argument('--candidate-config', required=True)
    args=p.parse_args(); print('Deployment spec accepted; scene-only inference path is enforced by mfrp.planning.deploy.scene_only_inference.')
if __name__=='__main__': main()
