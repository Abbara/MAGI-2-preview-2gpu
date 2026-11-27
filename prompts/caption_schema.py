"""Builders for MAGI-2's structured caption schema (assets/sample_enhanced_t2v.json):
a global_layer plus a dynamic_layer of timeline segments. Shared by the prompt
generators in prompts/eval/."""


def subj(sid, desc, pos, orient, **attrs):
    return {"subject_id": sid, "description": desc, "position": pos, "orientation": orient,
            "visual_attributes": attrs}


def obj(oid, desc, shape, texture, size, pos, orient="upright"):
    return {"object_id": oid, "description": desc, "shape_and_color": shape, "texture": texture,
            "relative_size": size, "position": pos, "orientation": orient}


def act(sid, ts, loc, size, orient, primary, body, motion, interaction="none", face="neutral", n=1):
    return {"subject_id": sid, "timestamp": ts, "number_of_subjects": n,
            "spatial_position": {"location": loc, "relative_size": size, "orientation": orient},
            "action": {"primary_action": primary, "body_configuration": body, "motion_detail": motion,
                       "interaction": interaction, "facial_expression": face}}


def seg(ts, desc, bg, cam, subjects=(), objects=(), events=(), lines=(), ambience=None, sfx=(), text=()):
    focus, position, composition, mtype, mdir, mint = cam
    return {
        "segment_basic_info": {
            "timestamp_range": ts, "segment_description": desc, "active_background": bg,
            "camera_delta": {"focus": focus, "camera_position": position, "composition": composition,
                             "camera_movement": {"type": mtype, "direction": mdir, "intensity": mint}}},
        "objects": list(objects), "alive_subjects": list(subjects), "text_render": list(text),
        "causal_events": [{"event_id": f"event {i+1}", "timestamp_range": e[0], "trigger": e[1],
                           "effect_outcome": e[2], "physics": e[3]} for i, e in enumerate(events)],
        "audio": {
            "dialogue_lines": [{"timestamp": l[0], "speaker": l[1], "text": l[2], "delivery": l[3]} for l in lines],
            "ambience_deltas": [{"timestamp_range": ts, "ambience": ambience}] if ambience else [],
            "special_audio_events": [{"timestamp": s[0], "sound_description": s[1], "visual_sync": s[2]} for s in sfx]},
    }


def prompt(context, description, aesthetics, ambience, language, speakers, lighting, background, camera,
           subjects=(), objects=(), segments=(), aspect_ratio="16:9"):
    style, contrast, saturation, colors, mood = aesthetics
    cond, direction, shadows = lighting
    cam_style, dof, lens = camera
    return {
        "global_layer": {
            "context": context, "description": description,
            "aesthetics": {"style": style, "contrast": contrast, "saturation": saturation,
                           "color_scheme": colors, "mood_atmosphere": mood},
            "audio_baseline": {"ambience": ambience,
                               "dialogue": {"language": language, "speaker_tags": list(speakers)}},
            "objects_static": list(objects),
            "video_metadata": {"duration": 10.0, "aspect_ratio": aspect_ratio},
            "lighting_baseline": {"conditions": cond, "direction": direction, "shadows": shadows,
                                  "source_consistency": "the lighting holds steady for the full clip"},
            "environment_baseline": {"background_setting": background},
            "alive_subjects_static": list(subjects),
            "camera_cinematography": {"overall_camera_style": cam_style, "default_depth_of_field": dof,
                                      "lens_focal_length": lens},
        },
        "dynamic_layer": {"timeline_segments": list(segments)},
    }
