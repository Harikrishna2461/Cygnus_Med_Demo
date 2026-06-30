# Guidance Video Assets

Place the guidance and ultrasound videos here, mirroring the QML asset paths.

Structure expected:
    guidance/sfj/standing_position.mp4
    guidance/sfj/observe_sfj.mp4
    guidance/sfj/sfj_groin.mp4
    guidance/sfj/probe_stable.mp4
    guidance/sfj/compress_calf.mp4
    guidance/sfj/us/sfj_right_leg.mp4
    guidance/sfj/us/bmode_sfj2.mp4
    guidance/sfj/us/colordoppler_sfj.mp4
    guidance/sfj/us/colordoppler_sfj2.mp4

    guidance/gsv_thigh/gsv_thigh.mp4
    guidance/gsv_thigh/pause_thigh.mp4
    guidance/gsv_thigh/compress_thigh.mp4
    guidance/gsv_thigh/us/bmode_gsv_proximal.mp4
    guidance/gsv_thigh/us/gsv_trunk.mp4
    guidance/gsv_thigh/us/colordoppler_gsv.mp4

    guidance/gsv_calf/knee_ankle.mp4
    guidance/gsv_calf/compress_calf.mp4
    guidance/gsv_calf/us/bmode_knee.mp4
    guidance/gsv_calf/us/bmode_calf.mp4
    guidance/gsv_calf/us/calf_longitudinal.mp4

    guidance/spj/patient_turn.mp4
    guidance/spj/scan_spj.mp4
    guidance/spj/compress_popliteal.mp4
    guidance/spj/us/colordoppler_popliteal.mp4
    guidance/spj/us/colordoppler_spj.mp4

    guidance/ssv/ssv.mp4
    guidance/ssv/scan_ssv.mp4
    guidance/ssv/compress_ssv.mp4
    guidance/ssv/scan_giacomini.mp4
    guidance/ssv/us/bmode_ssv.mp4
    guidance/ssv/us/colordoppler_ssv.mp4

The video_processor.py will:
  1. Concatenate the technique videos (excluding longitudinal-only clips)
  2. Extract frames at regular intervals
  3. Use those frames as mock ultrasound images aligned with scanner data
