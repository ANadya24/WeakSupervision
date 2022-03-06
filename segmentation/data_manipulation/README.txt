This folder cosists of scripts used for registration-based mask generation.

There are 2 options:
    1) There are cell images and some GT masks, you want to register cell masks from
    neighbouring frames by cropping it with fixed size and then add transformed mask to the frame where it was absent.
    First, you need to create crops using command:
    python3 create_crops_fixed_size.py
    (all the parameters are defined inside the python script)
    Second, you need to create masks from deformations using command:
    python3 create_masks_from_deforms.py create_config.json
    (all the parameters are defined in create_config.json)

    2) There are cell images and some GT masks, you want to register cell masks from
    neighbouring frames by cropping it with the size defined by cell mask size and then add transformed
    mask to the frame where it was absent.
    First, you need to create crops using command:
    python3 create_crops.py
    (all the parameters are defined inside the python script)
    Second, you need to create masks from deformations using command:
    python3 create_masks_from_deforms_expand_crops.py create_config_expand.json
    (all the parameters are defined in create_config.json)
