def pp_lang_by_stages_v1(obj_name, cont_name="container", rm_underscore=True):
    if rm_underscore:
        obj_name = obj_name.replace("_", " ")
        cont_name = cont_name.replace("_", " ")
    langs = [
        f'gripper open, reaching for {obj_name}, out of {cont_name}',
        f'gripper open, moving down over {obj_name}, out of {cont_name}',
        f'gripper closing, with {obj_name}, out of {cont_name}',
        f'gripper closed, moving up with {obj_name}, out of {cont_name}',
        f'gripper closed, moving sideways with {obj_name}, out of {cont_name}',
        f'gripper closed, with {obj_name}, above {cont_name}',
        f'gripper open, dropped {obj_name}, in {cont_name}'
    ]
    return langs


def pp_lang_by_stages_v1_rev(
        obj_name, cont_name="container", rm_underscore=True):
    if rm_underscore:
        obj_name = obj_name.replace("_", " ")
        cont_name = cont_name.replace("_", " ")
    langs = [
        f'gripper open, reaching for {obj_name}, in {cont_name}',
        f'gripper open, moving down over {obj_name}, in {cont_name}',
        f'gripper closing, with {obj_name}, in {cont_name}',
        f'gripper closed, moving up with {obj_name}, above {cont_name}',
        f'gripper closed, moving sideways with {obj_name}, out of {cont_name}',
        f'gripper closed, with {obj_name}, out of {cont_name}',
        f'gripper open, dropped {obj_name}, out of {cont_name}'
    ]
    return langs


def pp_real_lang_by_stages_v0(obj_name):
    """deprecated"""
    langs = [
        f"gripper moving toward {obj_name} outside container",
        f"gripper above {obj_name} outside container",
        f"gripper closing around {obj_name} outside container",
        f"gripper holding {obj_name} outside container and moving up",
        f"gripper holding {obj_name} moving toward container",
        f"gripper above {obj_name} in container",
    ]
    return langs


def pp_sim_lang_by_stages_v0(obj_name):
    """deprecated"""
    langs = [
        f"gripper moving toward {obj_name} outside container",
        f"gripper above {obj_name} outside container",
        f"gripper closing around {obj_name} outside container",
        f"gripper holding {obj_name} outside container and moving up",
        f"gripper holding {obj_name} moving toward container",
        f"gripper holding {obj_name} above container",
        f"gripper above {obj_name} in container",
    ]
    return langs


def ww_lang_by_stages(
        grasp_obj_name="last bead",
        flex_wraparound_obj_name="beads",
        central_obj_name="cylinder",
        rm_underscore=True):
    if rm_underscore:
        grasp_obj_name = grasp_obj_name.replace("_", " ")
        flex_wraparound_obj_name = flex_wraparound_obj_name.replace("_", " ")
        central_obj_name = central_obj_name.replace("_", " ")
    langs = [
        f"gripper open, reaching for {grasp_obj_name}",
        f"gripper open, moving down over {grasp_obj_name}",
        f"gripper closing around {grasp_obj_name}",
        f"gripper closed, moving up with {grasp_obj_name}",

        f"gripper closed, moving counterclockwise around {central_obj_name} with {flex_wraparound_obj_name} slightly wrapped",
        f"gripper closed, moving counterclockwise around {central_obj_name} with {flex_wraparound_obj_name} one quarter wrapped",
        f"gripper closed, moving counterclockwise around {central_obj_name} with {flex_wraparound_obj_name} half wrapped",
        f"gripper closed, moving counterclockwise around {central_obj_name} with {flex_wraparound_obj_name} three quarters wrapped",
        f"gripper closed, moving counterclockwise around {central_obj_name} with {flex_wraparound_obj_name} fully wrapped",

        f"gripper closed, moving clockwise around {central_obj_name} with {flex_wraparound_obj_name} slightly wrapped",
        f"gripper closed, moving clockwise around {central_obj_name} with {flex_wraparound_obj_name} one quarter wrapped",
        f"gripper closed, moving clockwise around {central_obj_name} with {flex_wraparound_obj_name} half wrapped",
        f"gripper closed, moving clockwise around {central_obj_name} with {flex_wraparound_obj_name} three quarters wrapped",
        f"gripper closed, moving clockwise around {central_obj_name} with {flex_wraparound_obj_name} fully wrapped",

        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully wrapped",
        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully unwrapped",
    ]
    return langs


def ww_lang_by_stages_debug(
        grasp_obj_name="last bead",
        flex_wraparound_obj_name="beads",
        central_obj_name="cylinder",
        rm_underscore=True):
    if rm_underscore:
        grasp_obj_name = grasp_obj_name.replace("_", " ")
        flex_wraparound_obj_name = flex_wraparound_obj_name.replace("_", " ")
        central_obj_name = central_obj_name.replace("_", " ")
    langs = [
        f"gripper open, reaching for {grasp_obj_name}",
        f"gripper open, moving down over {grasp_obj_name}",
        f"gripper closing around {grasp_obj_name}",
        f"gripper closed, moving up with {grasp_obj_name}",

        "counterclockwise slightly wrapped",
        "counterclockwise one quarter wrapped",
        "counterclockwise half wrapped",
        "counterclockwise three quarters wrapped",
        "counterclockwise fully wrapped",

        "clockwise slightly wrapped",
        "clockwise one quarter wrapped",
        "clockwise half wrapped",
        "clockwise three quarters wrapped",
        "clockwise fully wrapped",

        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully wrapped",
        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully unwrapped",
    ]
    return langs


def ww_lang_by_stages_v2(
        grasp_obj_name="last bead",
        flex_wraparound_obj_name="beads",
        central_obj_name="cylinder",
        rm_underscore=True):
    if rm_underscore:
        grasp_obj_name = grasp_obj_name.replace("_", " ")
        flex_wraparound_obj_name = flex_wraparound_obj_name.replace("_", " ")
        central_obj_name = central_obj_name.replace("_", " ")
    langs = [
        f"gripper open, reaching for {grasp_obj_name}",
        f"gripper open, moving down over {grasp_obj_name}",
        f"gripper closing around {grasp_obj_name}",
        f"gripper closed, moving up with {grasp_obj_name}",

        f"gripper closed, left of {central_obj_name}, moving counter-clockwise",
        f"gripper closed, in front of {central_obj_name}, moving counter-clockwise",
        f"gripper closed, right of {central_obj_name}, moving counter-clockwise",
        f"gripper closed, behind {central_obj_name}, moving counter-clockwise",

        f"gripper closed, left of {central_obj_name}, moving clockwise",
        f"gripper closed, in front of {central_obj_name}, moving clockwise",
        f"gripper closed, right of {central_obj_name}, moving clockwise",
        f"gripper closed, behind {central_obj_name}, moving clockwise",

        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully wrapped",
        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully unwrapped",
    ]
    return langs


def ww_lang_by_stages_simplified(
        grasp_obj_name="last bead",
        flex_wraparound_obj_name="beads",
        central_obj_name="cylinder",
        rm_underscore=True):
    if rm_underscore:
        grasp_obj_name = grasp_obj_name.replace("_", " ")
        flex_wraparound_obj_name = flex_wraparound_obj_name.replace("_", " ")
        central_obj_name = central_obj_name.replace("_", " ")
    langs = [
        f"gripper open, reaching for {grasp_obj_name}",
        f"gripper open, moving down over {grasp_obj_name}",
        f"gripper closing around {grasp_obj_name}",
        f"gripper closed, moving up with {grasp_obj_name}",

        "counter-clockwise left",
        "counter-clockwise front",
        "counter-clockwise right",
        "counter-clockwise back",

        "clockwise left",
        "clockwise front",
        "clockwise right",
        "clockwise back",

        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully wrapped",
        f"gripper open, above {grasp_obj_name} with {flex_wraparound_obj_name} fully unwrapped",
    ]
    return langs
