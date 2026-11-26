#!/usr/bin/env python3
"""Generate the capability-evaluation prompts (prompts/eval/NN_name.json).

Each prompt follows the structured caption schema of assets/sample_enhanced_t2v.json
(global_layer + dynamic_layer with two 5-second timeline segments). Run from the repo root:
    python3 prompts/eval/make_prompts.py
"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))


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
           subjects=(), objects=(), segments=()):
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
            "video_metadata": {"duration": 10.0, "aspect_ratio": "16:9"},
            "lighting_baseline": {"conditions": cond, "direction": direction, "shadows": shadows,
                                  "source_consistency": "the lighting holds steady for the full clip"},
            "environment_baseline": {"background_setting": background},
            "alive_subjects_static": list(subjects),
            "camera_cinematography": {"overall_camera_style": cam_style, "default_depth_of_field": dof,
                                      "lens_focal_length": lens},
        },
        "dynamic_layer": {"timeline_segments": list(segments)},
    }


A, B = "00:00.0-00:05.0", "00:05.0-00:10.0"
S1, S2, O1 = "<subject 1>", "<subject 2>", "<object 1>"
P = {}

# 01 talking head: lip sync, facial detail, English speech
P["01_talking_head"] = prompt(
    "A single continuous take of a woman speaking directly to camera in a quiet home studio.",
    "A woman in her thirties looks into the lens and delivers two short sentences about the weather, smiling at the end.",
    ("natural documentary interview look", "soft, gentle contrast", "natural skin tones, moderate saturation",
     "warm beige walls, a dark green sweater, soft daylight", "calm, friendly, intimate"),
    "a quiet room with a faint hum of distant traffic through a closed window", "English", [S1],
    ("soft daylight from a large window", "key light from frame left, gentle fill from the right", "soft, subtle shadows under the chin"),
    "a softly blurred home office with a bookshelf and a potted plant behind her",
    ("locked-off medium close-up on a tripod", "shallow, face sharp, background softly blurred", "50mm"),
    subjects=[subj(S1, "a woman in her thirties with shoulder-length dark brown hair", "centered in frame",
                   "facing the camera", age_group="early thirties", facial_features="brown eyes, light freckles, natural makeup",
                   clothing="dark green knit sweater", hair="shoulder-length dark brown hair, loosely tucked behind one ear")],
    segments=[
        seg(A, "She takes a small breath and begins speaking warmly to the camera.", "the room stays still",
            ("her eyes and mouth", "eye level, one and a half meters away", "medium close-up, head and shoulders centered", "static", "none", "none"),
            subjects=[act(S1, A, "center of frame", "head and shoulders fill the middle third", "facing camera",
                          "speaking to camera", "upright, shoulders relaxed", "lips form each word clearly, small natural head nods, occasional blink",
                          face="warm and engaged")],
            lines=[("00:00.6-00:04.2", S1, "The forecast says rain all week, but I don't believe it.", "conversational, warm, slightly amused")],
            ambience="quiet room tone"),
        seg(B, "She tilts her head, finishes her thought, and breaks into a broad smile.", "the room stays still",
            ("her face", "eye level", "medium close-up", "static", "none", "none"),
            subjects=[act(S1, B, "center of frame", "head and shoulders", "facing camera",
                          "finishing the sentence and smiling", "head tilts slightly to the right",
                          "lips articulate the words, then spread into a smile showing teeth, eyes crinkle",
                          face="playful, then smiling broadly")],
            lines=[("00:05.3-00:07.8", S1, "Look outside. It's perfectly sunny.", "light, playful, with a small laugh at the end")],
            sfx=[("00:08.2", "a short soft laugh", "matches her smile")], ambience="quiet room tone"),
    ])

# 02 two-person dialogue: turn-taking, two voices
P["02_two_person_dialogue"] = prompt(
    "A single continuous take in a cluttered woodworking shop where a grandfather and his grandson talk over a birdhouse.",
    "An elderly man holds up a small wooden birdhouse; the boy asks a question and the man answers, both smiling.",
    ("warm naturalistic drama", "moderate contrast", "warm, earthy saturation", "honey-colored wood, sawdust, a blue flannel shirt",
     "tender, cozy"),
    "a quiet workshop: a faint radio playing far away and birds outside", "English", [S1, S2],
    ("late afternoon sun through a dusty window", "from frame right, low and warm", "long soft shadows across the workbench"),
    "a wooden workbench covered in tools, wood shavings and clamps, pegboard wall behind",
    ("handheld-steady two-shot", "moderate; both faces sharp", "35mm"),
    subjects=[subj(S1, "an elderly man around seventy", "frame left behind the bench", "three-quarter toward the boy",
                   age_group="about seventy", facial_features="white beard, reading glasses on his nose, deep smile lines",
                   clothing="blue flannel shirt with rolled sleeves, a leather apron"),
              subj(S2, "a boy of about nine", "frame right", "three-quarter toward the man", age_group="about nine",
                   facial_features="messy light-brown hair, freckles", clothing="red hoodie")],
    objects=[obj(O1, "a small unpainted wooden birdhouse", "pale pine box with a peaked roof and a round hole",
                 "freshly sanded wood", "about the size of a loaf of bread", "in the man's hands at chest height")],
    segments=[
        seg(A, "The man lifts the birdhouse to show it; the boy leans in and asks a question.", "dust motes float in the sunbeam",
            ("both faces", "chest height, across the bench", "two-shot, man left, boy right", "slow push-in", "forward", "subtle"),
            subjects=[act(S1, A, "left third", "waist-up", "toward the boy", "presenting the birdhouse", "both hands raise the birdhouse",
                          "turns it slowly to show the round entry hole", "holds the birdhouse out to the boy", "proud"),
                      act(S2, A, "right third", "waist-up", "toward the man", "asking a question", "leans forward on the bench",
                          "lips move as he speaks, eyes on the birdhouse", "looks at the birdhouse", "curious")],
            lines=[("00:01.5-00:03.8", S2, "Grandpa, will a real bird actually live in there?", "eager, high child's voice")],
            ambience="workshop quiet, distant radio"),
        seg(B, "The man answers with a chuckle and hands the birdhouse to the boy, who grins.", "dust motes keep drifting",
            ("both faces", "chest height", "two-shot", "static", "none", "none"),
            subjects=[act(S1, B, "left third", "waist-up", "toward the boy", "answering and handing over the birdhouse",
                          "extends his arms", "speaks, then chuckles; places the birdhouse in the boy's hands", "hands over the birdhouse", "warm, amused"),
                      act(S2, B, "right third", "waist-up", "toward the man", "receiving the birdhouse", "takes it with both hands",
                          "grins widely and looks down at it", "holds the birdhouse", "delighted")],
            lines=[("00:05.3-00:08.0", S1, "If we build it right, they'll move in by spring.", "low, gravelly, kind, with a small chuckle")],
            sfx=[("00:08.6", "a soft wooden knock as the birdhouse changes hands", "matches the handover")],
            ambience="workshop quiet, birds outside"),
    ])

# 03 Mandarin speech in a market
P["03_mandarin_vendor"] = prompt(
    "A single continuous take at a busy night-market fruit stall.",
    "A cheerful fruit vendor holds up a mango and calls out a price to passing shoppers in Mandarin.",
    ("vibrant street documentary", "high contrast with glowing practical lights", "rich, saturated colors",
     "orange mangoes, red lanterns, green fruit crates under warm bulbs", "lively, bustling, friendly"),
    "a crowded night market: chatter, sizzling food stalls, scooters passing", "Mandarin Chinese", [S1],
    ("night, lit by hanging bare bulbs and red lanterns", "overhead warm practicals", "hard small shadows under the fruit"),
    "rows of fruit crates and a blurred crowd of shoppers under strings of lights",
    ("handheld medium shot", "shallow; vendor sharp, crowd blurred into bokeh", "50mm"),
    subjects=[subj(S1, "a middle-aged male fruit vendor", "center, behind the stall", "facing the camera",
                   age_group="around fifty", facial_features="short black hair, broad smile", clothing="white T-shirt and a navy apron")],
    objects=[obj(O1, "a ripe mango", "oval, orange-yellow with a red blush", "smooth waxy skin", "fits in his palm", "in his raised right hand")],
    segments=[
        seg(A, "The vendor lifts a mango toward the camera and calls out.", "shoppers stream past behind him",
            ("the vendor's face and the mango", "eye level across the stall", "medium shot, vendor centered", "handheld drift", "slight", "subtle"),
            subjects=[act(S1, A, "center", "waist-up", "facing camera", "calling out to shoppers", "right arm raised holding a mango",
                          "mouth wide as he calls out, eyebrows raised", "shows the mango to passersby", "enthusiastic")],
            lines=[("00:00.5-00:03.5", S1, "芒果又甜又新鲜，十块钱三个！", "loud, cheerful street-vendor call")],
            ambience="dense market chatter and sizzling woks"),
        seg(B, "He laughs, puts the mango in a paper bag and hands it across the stall.", "the crowd keeps moving",
            ("the vendor's hands and face", "eye level", "medium shot", "handheld drift", "slight", "subtle"),
            subjects=[act(S1, B, "center", "waist-up", "facing camera", "bagging the mango", "leans forward over the crates",
                          "drops the mango into a brown paper bag and holds it out", "hands the bag toward the camera", "laughing")],
            lines=[("00:05.5-00:07.5", S1, "来，尝尝看，不甜不要钱！", "warm, laughing, confident")],
            sfx=[("00:06.2", "crinkle of a paper bag", "matches the mango going into the bag")],
            ambience="market chatter, a scooter horn in the distance"),
    ])

# 04 street musician singing with guitar
P["04_street_guitarist"] = prompt(
    "A single continuous take of a busker playing guitar and singing on a cobblestone street at golden hour.",
    "A young man strums an acoustic guitar and sings a short gentle melody while a few people walk past.",
    ("warm cinematic street performance", "moderate contrast", "warm golden saturation",
     "golden sunlight, honey wood guitar, grey cobblestones", "nostalgic, gentle"),
    "a quiet old-town street with distant footsteps and pigeons", "English", [S1],
    ("golden hour sun low in the sky", "from behind frame left, rim-lighting his hair", "long warm shadows on the cobblestones"),
    "an old European street with stone facades, flower boxes and a café awning, softly blurred",
    ("slow dolly arc", "shallow; musician sharp", "85mm"),
    subjects=[subj(S1, "a young man in his twenties with curly hair", "center-left, sitting on a low stone step", "three-quarter toward camera",
                   age_group="mid-twenties", facial_features="curly dark hair, short beard", clothing="cream linen shirt, brown trousers")],
    objects=[obj(O1, "an acoustic guitar", "natural spruce top with a dark rosewood back", "glossy wood", "held across his lap",
                 "in his arms", "horizontal")],
    segments=[
        seg(A, "He strums a slow chord progression and starts singing.", "a couple walks past in the background",
            ("his face and strumming hand", "seated eye level", "medium shot", "dolly arc", "slowly from left to right", "gentle"),
            subjects=[act(S1, A, "center-left", "full upper body with guitar", "three-quarter", "playing guitar and singing",
                          "seated, guitar on knee, head tilted", "right hand strums in a steady rhythm, left hand changes chords on the neck, lips shape the lyric",
                          "plays the guitar", "eyes half closed, absorbed")],
            lines=[("00:00.8-00:04.6", S1, "Oh the evening light is falling slow...", "sung softly in a warm tenor, melodic")],
            sfx=[("00:00.0-00:05.0", "steady acoustic guitar strumming in a major key", "each strum matches the right hand")],
            ambience="soft street ambience under the music"),
        seg(B, "He sings the next line, looks up and smiles at a passerby while still strumming.", "a pigeon lands on the cobblestones",
            ("his face", "seated eye level", "medium close-up", "dolly arc", "continuing left to right", "gentle"),
            subjects=[act(S1, B, "center", "upper body", "three-quarter", "singing and smiling", "lifts his head",
                          "keeps strumming, finishes the line, then smiles", "plays the guitar", "warm smile")],
            lines=[("00:05.2-00:08.6", S1, "and I'll be waiting where the rivers go.", "sung softly, holding the last note")],
            sfx=[("00:05.0-00:10.0", "the guitar strumming continues and ends on a ringing chord", "matches the final strum")],
            ambience="street ambience, distant church bell"),
    ])

# 05 piano hands: fine finger motion + music
P["05_piano_hands"] = prompt(
    "A single continuous macro-style take of a pianist's hands on a grand piano.",
    "Close-up of two hands playing a flowing arpeggio on the keys, then a soft final chord.",
    ("elegant concert cinematography", "high contrast, deep blacks", "low saturation",
     "black lacquer, ivory keys, warm skin tones", "intimate, refined"),
    "a quiet concert hall with a slight natural reverb", "none", [],
    ("a single warm spotlight from above", "top-down, slightly from the right", "crisp shadows of the fingers on the keys"),
    "the black lacquered body of a grand piano fading into darkness",
    ("slow lateral slide along the keyboard", "very shallow; the fingers sharp, keys falling off into blur", "100mm macro"),
    subjects=[subj(S1, "the hands of an adult pianist", "on the keyboard, filling the lower two thirds", "fingers pointing away from camera",
                   hands="slender fingers, short clean nails, a thin silver ring on the right hand")],
    segments=[
        seg(A, "The right hand plays a rising arpeggio while the left holds a low chord.", "the piano's reflection shimmers",
            ("the right hand's fingers", "low, at keyboard height", "close-up, keys diagonal across the frame", "lateral slide", "left to right", "slow"),
            subjects=[act(S1, A, "center", "two hands fill the frame", "fingers away from camera", "playing an arpeggio",
                          "wrists level, fingers curved", "fingers press keys one after another in a smooth wave; keys visibly depress and rise",
                          "fingers on the keys")],
            sfx=[("00:00.0-00:05.0", "a flowing rising piano arpeggio in C major with natural hall reverb", "each note matches a key going down")],
            ambience="silent hall"),
        seg(B, "Both hands come together for a soft final chord and lift slowly from the keys.", "reflections settle",
            ("both hands", "keyboard height", "close-up", "lateral slide", "slowing to a stop", "very slow"),
            subjects=[act(S1, B, "center", "both hands", "fingers away from camera", "playing a final chord and lifting",
                          "hands press together, then lift", "all fingers press at once, hold, then rise gently off the keys", "fingers on the keys")],
            sfx=[("00:06.0", "a soft sustained major chord", "matches both hands pressing down"),
                 ("00:06.0-00:10.0", "the chord rings and fades into the hall reverb", "matches the hands lifting")],
            ambience="silent hall"),
    ])

# 06 skateboard: fast action, body motion
P["06_skateboard_kickflip"] = prompt(
    "A single continuous take at an outdoor concrete skatepark on a sunny afternoon.",
    "A skateboarder rolls toward the camera, pops a kickflip, lands it cleanly and rolls away.",
    ("energetic action sports film", "high contrast in harsh sun", "vivid saturation",
     "pale grey concrete, bright blue sky, a yellow T-shirt", "energetic, exciting"),
    "a skatepark: distant wheels rolling, wind, a few people cheering far away", "none", [],
    ("bright midday sun", "high and slightly behind the skater", "short hard shadows directly under him"),
    "a concrete skatepark with ramps, rails and palm trees against a clear sky",
    ("low-angle tracking shot", "deep; skater and park both in focus", "24mm wide"),
    subjects=[subj(S1, "a young male skateboarder", "rolling toward camera", "facing camera",
                   age_group="about twenty", clothing="yellow T-shirt, black cargo pants, white sneakers, backward cap")],
    objects=[obj(O1, "a skateboard", "black grip tape on top, red graphic on the bottom", "sandpaper grip, scuffed wood",
                 "about knee-length", "under the skater's feet", "horizontal")],
    segments=[
        seg(A, "He pushes twice, crouches and pops a kickflip; the board spins under him.", "palm fronds sway",
            ("the skater and board", "low, near the ground", "full body, skater centered", "tracking backward", "backward as he approaches", "fast"),
            subjects=[act(S1, A, "center", "full body fills half the frame height", "facing camera",
                          "pushing then kickflipping", "crouches low, then jumps", "back foot snaps the tail, front foot flicks off the edge, the board flips a full rotation in the air",
                          "jumps over the spinning board", "focused")],
            events=[("00:03.2-00:04.2", "his front foot flicks the board", "the board rotates one full turn under him", "the board spins along its long axis then is caught by both feet")],
            sfx=[("00:00.5-00:02.5", "two pushes of urethane wheels rolling on concrete", "matches the pushes"),
                 ("00:03.2", "a sharp wooden pop of the tail hitting concrete", "matches the pop")],
            ambience="wind and distant wheels"),
        seg(B, "He lands with both feet on the bolts, rolls past the camera and raises a fist.", "two friends cheer in the background",
            ("the skater", "low", "full body, then passing to the right", "pan", "following him to the right", "fast"),
            subjects=[act(S1, B, "center then exiting right", "full body", "turning to profile", "landing and rolling away",
                          "knees bend on impact, then he stands tall", "lands with a slight crouch, rolls past, lifts his right fist",
                          "rides the board", "triumphant grin")],
            sfx=[("00:05.1", "a solid clack as all four wheels land", "matches the landing"),
                 ("00:06.0-00:08.0", "friends whoop and cheer", "matches the raised fist")],
            ambience="wind and wheels fading"),
    ])

# 07 liquid physics: coffee pour
P["07_coffee_pour"] = prompt(
    "A single continuous close-up in a bright kitchen of hot coffee being poured into a glass mug.",
    "Coffee streams from a gooseneck kettle-style server into a clear glass mug, swirling, then cream is poured in and blooms.",
    ("commercial food cinematography", "crisp, clean contrast", "rich warm browns, otherwise neutral",
     "dark coffee, white cream, clear glass, pale marble counter", "calm, appetizing"),
    "a quiet kitchen in the morning, a clock ticking softly", "none", [],
    ("soft morning window light", "from behind the mug, backlighting the liquid", "soft shadows on the marble"),
    "a white marble countertop and a blurred bright kitchen window",
    ("locked-off close-up", "shallow; the mug sharp", "90mm"),
    objects=[obj(O1, "a clear double-wall glass mug", "transparent tall cylinder", "smooth glass with gentle reflections",
                 "fills the middle third of the frame", "center on the counter")],
    segments=[
        seg(A, "A thin stream of dark coffee pours into the empty glass mug and fills it two thirds.", "steam begins to rise",
            ("the stream and the liquid surface", "at mug height", "close-up, mug centered", "static", "none", "none"),
            events=[("00:00.5-00:05.0", "coffee pours into the mug", "the level rises with swirling foam and small bubbles",
                     "the liquid swirls in a vortex, a thin crema forms at the top, steam lifts")],
            sfx=[("00:00.5-00:05.0", "a gentle liquid pouring sound rising in pitch as the mug fills", "matches the stream")],
            ambience="soft clock ticking"),
        seg(B, "Cream pours in from above and blooms into the coffee in slow white clouds.", "steam keeps rising",
            ("the clouds of cream", "at mug height", "close-up", "slow push-in", "forward", "very slow"),
            events=[("00:05.5-00:10.0", "cream is poured in", "white plumes billow and curl downward through the dark coffee",
                     "the denser cream sinks and swirls into marbled turbulent clouds")],
            sfx=[("00:05.5-00:07.0", "a soft trickle of cream", "matches the pour")],
            ambience="soft clock ticking"),
    ])

# 08 fire and sparks
P["08_campfire_night"] = prompt(
    "A single continuous take of a campfire at night in a forest clearing.",
    "Flames dance over crackling logs; someone drops a new log on and a burst of sparks spirals up into the dark.",
    ("moody naturalistic night cinematography", "very high contrast", "deep oranges, otherwise dark",
     "orange flame, black night, glowing red embers", "warm, primal, quiet"),
    "night forest: crickets, a soft breeze, the crackle of the fire", "none", [],
    ("firelight is the only source", "from the fire, flickering upward", "deep moving shadows"),
    "dark tree trunks barely lit by the fire, a starry sky above",
    ("locked-off low shot", "moderate; fire sharp", "35mm"),
    objects=[obj(O1, "a campfire in a ring of stones", "stacked logs with bright orange flames", "charred bark, glowing embers",
                 "fills the lower half of the frame", "center")],
    segments=[
        seg(A, "The fire burns steadily, flames licking upward and embers pulsing.", "smoke drifts upward into the dark",
            ("the flames", "low, at ground level", "medium close-up, fire centered", "static", "none", "none"),
            events=[("00:01.0-00:05.0", "the fire burns", "flames flicker and twist, a log shifts", "turbulent flames, glowing coals pulse with heat")],
            sfx=[("00:00.0-00:05.0", "a steady crackle and pop of burning wood", "matches small sparks popping")],
            ambience="crickets and soft breeze"),
        seg(B, "A hand drops a fresh log onto the fire and a column of sparks erupts upward.", "sparks rise into the starry sky",
            ("the sparks", "low", "medium close-up, tilting up slightly", "tilt", "upward following the sparks", "slow"),
            events=[("00:05.5-00:07.0", "a log is dropped onto the coals", "a shower of bright sparks spirals upward",
                     "the impact releases embers that rise on the hot air and fade")],
            sfx=[("00:05.6", "a heavy thud of the log and a whoosh of flame", "matches the log landing"),
                 ("00:05.8-00:08.0", "a burst of crackling pops", "matches the sparks")],
            ambience="crickets continue"),
    ])

# 09 crowd + rain + neon
P["09_tokyo_crosswalk_rain"] = prompt(
    "A single continuous take of a busy pedestrian crossing in a big city at night in the rain.",
    "Dozens of people with umbrellas cross a wide intersection as neon signs reflect on the wet asphalt.",
    ("cinematic urban night photography", "high contrast", "saturated neon colors",
     "magenta and cyan neon, black wet asphalt, clear and colored umbrellas", "busy, atmospheric"),
    "steady rain, crosswalk beeping signal, traffic hiss, crowd footsteps", "none", [],
    ("night, lit by neon signs and street lights", "from the signs on the buildings", "glossy reflections instead of shadows"),
    "tall buildings covered in glowing signs and billboards, a wide zebra crossing",
    ("slow high-angle crane down", "deep; the whole crossing in focus", "35mm"),
    subjects=[subj(S1, "a crowd of pedestrians with umbrellas", "spread across the crosswalk", "walking in many directions",
                   clothing="dark coats, a few bright raincoats")],
    segments=[
        seg(A, "The signal turns and the crowd floods into the crossing from all sides.", "rain streaks through the neon light",
            ("the crowd", "high above the crossing", "wide shot, crossing fills the frame", "crane", "slowly descending", "slow"),
            subjects=[act(S1, A, "everywhere across the crossing", "small figures", "all directions", "crossing the street",
                          "walking briskly under umbrellas", "hundreds of footsteps splash through puddles, umbrellas bob", "weave past each other", n=50)],
            sfx=[("00:00.5-00:05.0", "the electronic crosswalk melody beeping", "matches the signal turning green")],
            ambience="heavy rain and crowd footsteps"),
        seg(B, "The camera settles at street level as a woman with a clear umbrella walks toward it, neon reflected in the umbrella.", "puddles ripple",
            ("a woman with a clear umbrella", "eye level", "medium-wide, crowd behind her", "crane", "settling to a stop", "slow"),
            subjects=[act(S1, B, "center foreground and behind", "one figure larger in the foreground", "toward camera", "walking toward the camera",
                          "upright, umbrella held high", "raindrops bead and run down the clear umbrella, neon colors slide across it", "walks through the crowd", "calm")],
            sfx=[("00:06.0", "a car tire hissing through water", "matches a taxi passing behind")],
            ambience="rain on umbrellas, close and crisp"),
    ])

# 10 text rendering: chalkboard lettering
P["10_chalkboard_lettering"] = prompt(
    "A single continuous take of a bakery owner hand-lettering a chalkboard menu.",
    "A woman writes the words 'FRESH BREAD' in large white chalk capital letters on a black chalkboard, then steps back.",
    ("warm small-business documentary", "moderate contrast", "warm natural saturation",
     "black chalkboard, white chalk, warm wood frame, flour-dusted apron", "pleasant, crafty"),
    "a quiet bakery in the early morning, an oven fan humming", "none", [],
    ("warm morning light from the shop window", "from frame right", "soft shadows"),
    "a bakery counter with baskets of bread loaves softly blurred behind the board",
    ("locked-off medium shot", "moderate; the board sharp", "50mm"),
    subjects=[subj(S1, "a woman in her forties, the baker", "frame right of the chalkboard", "profile facing the board",
                   age_group="forties", facial_features="hair tied up in a bun", clothing="white shirt and a flour-dusted denim apron")],
    objects=[obj(O1, "a large black chalkboard on an easel", "rectangular black board in a light wooden frame",
                 "matte slate with faint old chalk smudges", "taller than her torso", "center-left")],
    segments=[
        seg(A, "She writes 'FRESH' letter by letter in bold white chalk capitals.", "steam from the ovens drifts in the background",
            ("the chalk and the letters", "chest height", "medium shot, board left, baker right", "static", "none", "none"),
            subjects=[act(S1, A, "right third", "waist-up", "profile", "writing on the chalkboard", "right arm raised to the board",
                          "the chalk draws each stroke of F, R, E, S, H in sequence", "writes on the board", "concentrated")],
            text=[{"timestamp_range": "00:00.5-00:05.0", "content": "FRESH", "style": "bold hand-drawn white chalk capital letters",
                   "position": "upper half of the chalkboard, appearing letter by letter"}],
            sfx=[("00:00.5-00:05.0", "the scratchy tapping of chalk on slate", "each stroke matches the chalk moving")],
            ambience="oven fan hum"),
        seg(B, "She writes 'BREAD' on the line below, then steps back and dusts off her hands.", "bread loaves on the counter",
            ("the finished words", "chest height", "medium shot", "static", "none", "none"),
            subjects=[act(S1, B, "right third", "waist-up", "profile, then turning toward camera", "finishing the word and stepping back",
                          "lowers her arm, steps back", "writes B, R, E, A, D, then claps chalk dust off her hands", "admires the board", "satisfied smile")],
            text=[{"timestamp_range": "00:05.0-00:10.0", "content": "FRESH BREAD", "style": "bold hand-drawn white chalk capital letters, two lines",
                   "position": "FRESH on the top line, BREAD below it, centered on the board"}],
            sfx=[("00:05.0-00:08.0", "chalk scratching on slate", "matches the writing"),
                 ("00:08.8", "a soft clap of dusty hands", "matches the hand clap")],
            ambience="oven fan hum"),
    ])

# 11 aerial: drone following a car
P["11_drone_coastal_car"] = prompt(
    "A single continuous aerial take following a car along a winding coastal cliff road.",
    "A red vintage convertible drives along a serpentine road carved into green cliffs above a turquoise sea; the drone follows and rises.",
    ("epic travel commercial aerial cinematography", "high contrast, crisp", "vivid saturation",
     "turquoise sea, white surf, green cliffs, a red car", "free, adventurous"),
    "high wind and the distant hum of the car engine, waves crashing far below", "none", [],
    ("bright late-morning sun", "from frame left", "crisp shadows of the cliffs on the water"),
    "a rugged coastline with steep green cliffs, a winding road, and a deep blue sea",
    ("smooth drone tracking shot", "deep; everything sharp", "24mm aerial"),
    objects=[obj(O1, "a red vintage convertible", "glossy cherry-red open-top 1960s convertible", "polished paint, chrome trim",
                 "small in the landscape", "on the road in the center", "driving along the road")],
    segments=[
        seg(A, "The drone follows behind the car as it rounds a hairpin bend above the sea.", "waves break white against the rocks",
            ("the red car", "thirty meters above and behind the car", "wide, car in the center third", "tracking", "forward along the road", "smooth"),
            objects=[{"object_id": O1, "timestamp": A, "number_of_objects": 1,
                      "dynamic_state": {"state_change": "the car takes a hairpin turn", "motion_detail": "it leans slightly into the bend"},
                      "spatial_position": {"location": "center", "relative_size": "small", "orientation": "heading away from camera"}}],
            sfx=[("00:00.0-00:05.0", "a smooth engine hum rising and falling with the turn", "matches the car's motion")],
            ambience="wind and distant surf"),
        seg(B, "The drone rises and tilts down to reveal the whole coastline as the car continues on.", "the road winds into the distance",
            ("the coastline", "climbing to eighty meters", "very wide, car tiny on the road", "crane up", "rising and tilting down", "slow"),
            objects=[{"object_id": O1, "timestamp": B, "number_of_objects": 1,
                      "dynamic_state": {"state_change": "the car drives on", "motion_detail": "it shrinks as the camera rises"},
                      "spatial_position": {"location": "lower center", "relative_size": "tiny", "orientation": "driving along the coast"}}],
            sfx=[("00:05.0-00:10.0", "the engine fades under rising wind", "matches the camera rising")],
            ambience="wind and surf"),
    ])

# 12 food + sizzle audio
P["12_steak_sizzle"] = prompt(
    "A single continuous close-up of a steak being butter-basted in a cast-iron pan.",
    "A thick steak sizzles in a black skillet; a cook tilts the pan and spoons foaming butter with garlic and thyme over it.",
    ("high-end food cinematography", "high contrast, glossy highlights", "warm, rich saturation",
     "deep brown crust, golden foaming butter, green thyme, black iron", "indulgent, mouthwatering"),
    "a restaurant kitchen: a loud steady sizzle and an extractor fan", "none", [],
    ("warm overhead kitchen light", "top-down from the left", "deep shadows inside the pan"),
    "a dark steel stovetop with blue gas flames under the skillet",
    ("locked-off close-up at a low angle", "shallow; the steak sharp", "85mm"),
    subjects=[subj(S1, "a cook's hands", "entering from the top of frame", "toward the pan", hands="tanned hands, one holding a large spoon")],
    objects=[obj(O1, "a thick ribeye steak in a cast-iron skillet", "dark caramelized crust with pink edges", "crisp seared surface, glistening fat",
                 "fills most of the pan", "center")],
    segments=[
        seg(A, "The steak sizzles hard; a knob of butter, garlic and thyme are dropped in and begin to foam.", "smoke curls up from the pan",
            ("the steak's crust", "low, at pan height", "close-up", "slow push-in", "forward", "subtle"),
            events=[("00:01.5-00:05.0", "butter hits the hot pan", "it melts instantly and foams around the steak", "bubbling, spattering fat, wisps of smoke")],
            sfx=[("00:00.0-00:05.0", "a loud continuous sizzle that intensifies when the butter drops in", "matches the butter landing at 00:01.5")],
            ambience="extractor fan"),
        seg(B, "A hand tilts the pan and repeatedly spoons foaming butter over the steak.", "smoke drifts through the light",
            ("the spoon and the steak", "pan height", "close-up", "static", "none", "none"),
            subjects=[act(S1, B, "top of frame", "hands only", "toward the pan", "basting", "one hand tilts the handle, the other spoons",
                          "the spoon scoops foaming butter and pours it over the crust three times", "bastes the steak")],
            events=[("00:05.5-00:10.0", "butter is spooned over the steak", "it runs down the crust and sizzles", "hot fat crackles and spits")],
            sfx=[("00:05.5-00:10.0", "rhythmic sizzle bursts with each spoonful", "matches each pour of butter")],
            ambience="extractor fan"),
    ])

# 13 weather: thunderstorm with synced thunder
P["13_thunderstorm_field"] = prompt(
    "A single continuous wide take of a thunderstorm rolling over a wheat field at dusk.",
    "Dark storm clouds churn over a golden wheat field bending in the wind; a lightning bolt strikes in the distance followed by thunder.",
    ("dramatic landscape cinematography", "very high contrast", "moody, desaturated with golden wheat",
     "slate-blue clouds, golden wheat, a flash of white lightning", "ominous, powerful"),
    "strong wind rushing through the wheat, distant rumbling thunder", "none", [],
    ("dusk under heavy storm clouds, a band of light at the horizon", "low from the horizon", "no hard shadows, flat stormy light"),
    "an endless field of ripe wheat under a vast sky of rolling storm clouds, a lone tree on the horizon",
    ("locked-off wide shot on a tripod", "deep; everything sharp", "24mm"),
    segments=[
        seg(A, "Wind sweeps waves across the wheat as the clouds roll toward the camera.", "the clouds boil and churn",
            ("the sky and the field", "eye level, standing in the field", "wide, horizon in the lower third", "static", "none", "none"),
            events=[("00:00.0-00:05.0", "strong gusts of wind", "waves ripple across the wheat field", "stalks bend in rolling waves, clouds move fast")],
            sfx=[("00:02.0", "a low distant rumble of thunder", "the clouds flicker faintly")],
            ambience="rushing wind through wheat"),
        seg(B, "A bright forked lightning bolt strikes near the lone tree; thunder cracks a moment later and rain begins.", "rain curtains sweep in",
            ("the lightning", "eye level", "wide", "static", "none", "none"),
            events=[("00:06.0-00:06.4", "lightning strikes near the tree", "the whole landscape flashes white for an instant",
                     "a branching bolt connects cloud to ground"),
                    ("00:07.5-00:10.0", "the rain front arrives", "grey rain sheets sweep across the field", "rain falls at an angle in the wind")],
            sfx=[("00:06.9", "a sharp, deafening crack of thunder rolling into a long rumble", "follows the lightning flash after a short delay"),
                 ("00:07.5-00:10.0", "heavy rain hissing across the wheat", "matches the rain front")],
            ambience="wind and rain"),
    ])

for name, data in P.items():
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
print(f"wrote {len(P)} prompts to {OUT}")
